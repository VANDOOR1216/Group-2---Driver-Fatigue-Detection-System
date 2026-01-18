# fatigue/state_tracker.py
from collections import deque
import numpy as np
from enum import Enum
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'fatigue_detection'))
from config import CONFIG


class FatigueState(Enum):
    """Fatigue State Enum"""
    NORMAL = "Normal"
    MILD = "Mild Fatigue"
    SEVERE = "Severe Fatigue"


class FatigueTracker:
    """疲劳状态跟踪器"""
    
    def __init__(self, window_size=None):
        """
        初始化疲劳跟踪器
        
        Args:
            window_size: 时间窗口大小（帧数）
        """
        self.window_size = window_size or CONFIG['fatigue']['window_size']
        
        # 数据缓冲区
        self.ear_buffer = deque(maxlen=self.window_size)
        self.mar_buffer = deque(maxlen=self.window_size)
        
        # 状态变量
        self.current_state = FatigueState.NORMAL
        self.last_state = FatigueState.NORMAL
        
        # 统计计数器
        self.blink_count = 0
        self.yawn_count = 0
        self.frame_count = 0
        
        # 连续闭眼帧数
        self.consecutive_closed_frames = 0
        self.max_consecutive_closed = 0
        # 连续打哈欠帧数
        self.consecutive_yawn_frames = 0
        
        # 配置阈值
        self.ear_threshold = CONFIG['ear']['threshold']
        self.mar_threshold = CONFIG['mar']['threshold']
        self.perclos_mild = CONFIG['fatigue']['perclos_mild']
        self.perclos_severe = CONFIG['fatigue']['perclos_severe']
        self.yawn_alarm = CONFIG['fatigue']['yawn_threshold']
        self.continuous_blink = CONFIG['fatigue']['continuous_blink']
        self.max_closed_frames_threshold = CONFIG['fatigue'].get('max_closed_frames', 15)
        
        # 状态标志
        self.is_blinking = False
        self.is_yawning = False
        
        # 历史数据
        self.state_history = []
        
        # 事件历史（用于计算近期频率）
        self.blink_history = deque()
        self.yawn_history = deque()
        self.history_window = 900  # 30秒历史窗口 (假设30fps)
    
    def update(self, ear, mar):
        """
        更新疲劳状态
        
        Args:
            ear: 平均EAR值
            mar: MAR值
            
        Returns:
            tuple: (当前状态, 统计信息)
        """
        self.frame_count += 1
        
        # 清理过期的事件历史
        while self.blink_history and self.blink_history[0] < self.frame_count - self.history_window:
            self.blink_history.popleft()
        while self.yawn_history and self.yawn_history[0] < self.frame_count - self.history_window:
            self.yawn_history.popleft()
            
        # 更新计数
        self.blink_count = len(self.blink_history)
        self.yawn_count = len(self.yawn_history)
        
        # 添加到缓冲区
        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)
        
        # 更新眨眼状态
        self._update_blink_state(ear)
        
        # 更新打哈欠状态
        self._update_yawn_state(mar)
        
        # 计算当前窗口内的最大连续闭眼帧数
        self._update_max_consecutive_closed()
        
        # 计算PERCLOS
        perclos = self._calculate_perclos()
        
        # 判断疲劳状态
        self.last_state = self.current_state
        self.current_state = self._determine_fatigue_state(perclos)
        
        # 记录状态历史
        self.state_history.append(self.current_state)
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        # 获取统计信息
        stats = self.get_statistics(perclos)
        
        return self.current_state, stats
    
    def _update_blink_state(self, ear):
        """更新眨眼状态"""
        if ear < self.ear_threshold:
            self.consecutive_closed_frames += 1
            
            # 检测眨眼（从睁眼到闭眼）
            if not self.is_blinking:
                self.is_blinking = True
                
                # 如果闭眼超过一定帧数，计数为有效眨眼
                if self.consecutive_closed_frames >= CONFIG['ear']['frame_buffer']:
                    self.blink_history.append(self.frame_count)
        else:
            self.consecutive_closed_frames = 0
            self.is_blinking = False
    
    def _update_yawn_state(self, mar):
        """更新打哈欠状态"""
        # 增加防抖机制：只有连续多帧张嘴才算哈欠
        yawn_frames_threshold = 5  # 至少连续5帧（约0.15秒）
        
        if mar > self.mar_threshold:
            self.consecutive_yawn_frames += 1
            
            if self.consecutive_yawn_frames >= yawn_frames_threshold:
                if not self.is_yawning:
                    self.is_yawning = True
                    self.yawn_history.append(self.frame_count)
        else:
            self.consecutive_yawn_frames = 0
            self.is_yawning = False

    def _update_max_consecutive_closed(self):
        """计算当前缓冲区内的最大连续闭眼帧数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for ear in self.ear_buffer:
            if ear < self.ear_threshold:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 0
        
        # 检查最后一个序列
        max_consecutive = max(max_consecutive, current_consecutive)
        
        self.max_consecutive_closed = max_consecutive
    
    def _calculate_perclos(self):
        """计算PERCLOS值"""
        if len(self.ear_buffer) == 0:
            return 0.0
        
        # 统计闭眼帧数
        closed_frames = sum(1 for ear in self.ear_buffer if ear < self.ear_threshold)
        
        # 计算百分比
        perclos = closed_frames / len(self.ear_buffer)
        
        return perclos
    
    def _determine_fatigue_state(self, perclos):
        """根据PERCLOS值确定疲劳状态"""
        
        # 迟滞阈值 (Hysteresis)
        # 防止状态在阈值附近抖动
        hysteresis = 0.10
        
        severe_th = self.perclos_severe
        mild_th = self.perclos_mild
        
        # 如果当前是疲劳状态，降低退出的门槛（增加粘性，防止回跳）
        if self.current_state == FatigueState.SEVERE:
            severe_th -= hysteresis
        elif self.current_state == FatigueState.MILD:
            mild_th -= hysteresis
            
        # 规则1: PERCLOS标准
        if perclos >= severe_th:
            return FatigueState.SEVERE
        elif perclos >= mild_th:
            return FatigueState.MILD
        
        # 规则2: 连续打哈欠
        if self.yawn_count >= self.yawn_alarm:
            return FatigueState.SEVERE
        
        # 规则3: 频繁眨眼
        if self.blink_count >= self.continuous_blink * (self.window_size / 30):
            return FatigueState.MILD
        
        # 规则4: 长时间闭眼
        if self.max_consecutive_closed > self.max_closed_frames_threshold:
            return FatigueState.SEVERE
        
        return FatigueState.NORMAL
    
    def get_statistics(self, perclos=None):
        """获取统计信息"""
        if perclos is None:
            perclos = self._calculate_perclos()
        
        avg_ear = np.mean(self.ear_buffer) if self.ear_buffer else 0.0
        avg_mar = np.mean(self.mar_buffer) if self.mar_buffer else 0.0
        
        return {
            'perclos': perclos,
            'avg_ear': avg_ear,
            'avg_mar': avg_mar,
            'blink_count': self.blink_count,
            'yawn_count': self.yawn_count,
            'frame_count': self.frame_count,
            'consecutive_closed': self.consecutive_closed_frames,
            'max_consecutive_closed': self.max_consecutive_closed,
            'current_state': self.current_state.value,
            'last_state': self.last_state.value
        }
    
    def reset(self):
        """重置跟踪器状态"""
        self.ear_buffer.clear()
        self.mar_buffer.clear()
        self.blink_count = 0
        self.yawn_count = 0
        self.consecutive_closed_frames = 0
        self.max_consecutive_closed = 0
        self.current_state = FatigueState.NORMAL
        self.last_state = FatigueState.NORMAL
    
    def get_state_transitions(self):
        """获取状态转移历史"""
        transitions = []
        for i in range(1, len(self.state_history)):
            if self.state_history[i] != self.state_history[i-1]:
                transitions.append({
                    'frame': i,
                    'from': self.state_history[i-1].value,
                    'to': self.state_history[i].value
                })
        return transitions