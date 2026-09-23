"""风险阈值的唯一事实来源与包含关系。

超时签到风险（overdue_minutes 为已完成整分钟数，
换算规则见 trailforge.timekeeping.whole_minutes_late）：

    [0, 30)     → LOW
    [30, 120)   → MODERATE   （恰好 30 分钟即进入 MODERATE）
    [120, 360)  → HIGH       （恰好 120 分钟即进入 HIGH）
    [360, ∞)    → CRITICAL   （恰好 360 分钟即进入 CRITICAL）

风险评估分值（score = likelihood × impact，范围 1..25）：

    [1, 4]   → LOW
    [5, 9]   → MODERATE
    [10, 16] → HIGH
    [17, 25] → CRITICAL

两套阈值都是左闭右开（分钟）/ 闭区间（分值），边界值归入更高一档。
"""

from __future__ import annotations

from trailforge.domain.enums import RiskLevel

#: 超时签到风险阈值（分钟，含下界）。
OVERDUE_MODERATE_MINUTES = 30
OVERDUE_HIGH_MINUTES = 120
OVERDUE_CRITICAL_MINUTES = 360

#: 风险评估分值阈值（含上界）。
SCORE_LOW_MAX = 4
SCORE_MODERATE_MAX = 9
SCORE_HIGH_MAX = 16


def overdue_risk_level(overdue_minutes: int) -> RiskLevel:
    """按已完成整分钟数判定超时签到风险等级（边界值进入更高一档）。"""
    if overdue_minutes >= OVERDUE_CRITICAL_MINUTES:
        return RiskLevel.CRITICAL
    if overdue_minutes >= OVERDUE_HIGH_MINUTES:
        return RiskLevel.HIGH
    if overdue_minutes >= OVERDUE_MODERATE_MINUTES:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def score_risk_level(score: int) -> RiskLevel:
    """按 likelihood × impact 分值判定风险评估等级（边界值留在本档）。"""
    if score <= SCORE_LOW_MAX:
        return RiskLevel.LOW
    if score <= SCORE_MODERATE_MAX:
        return RiskLevel.MODERATE
    if score <= SCORE_HIGH_MAX:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL
