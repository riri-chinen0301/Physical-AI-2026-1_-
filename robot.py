"""Planar 2R arm, analytic inverse kinematics and joint interpolation."""
# このファイル：アームの関節角と手先の位置を計算します。
# 座標の単位はこの2D教材内のピクセル、角度の単位はラジアンです。
import math
import numpy as np

# アームの土台の位置と、2本のリンク（腕の棒）の長さです。
BASE = np.array([400., 420.])
L1, L2 = 255., 235.
# 分類先の箱の中心位置です。A・B・Cで行き先を変えます。
BINS = {'A':np.array([655., 105.]), 'B':np.array([655., 235.]), 'C':np.array([655., 365.])}
# 1個の仕分けが終わるたびに戻る待機位置です。
HOME = np.array([425., 230.])

# 逆運動学（IK）：動かしたい手先の位置から、肩と肘の角度を求めます。
def ik(point):
    # 土台を原点とした相対座標に変えます。
    x,y = np.asarray(point)-BASE
    # 余弦定理で肘の角度のcos値を求めます。
    c = (x*x+y*y-L1*L1-L2*L2)/(2*L1*L2)
    # cos値が範囲外ならアームが届かないため、無理に動かさずエラーにします。
    if not -1 <= c <= 1: raise ValueError(f'Unreachable target {point}')
    # 肘の曲げ方は2通りありますが、この教材では負の角度になる側を選びます。
    q2 = -math.acos(c)
    # 肘の角度を考慮して肩の角度を計算します。
    q1 = math.atan2(y,x)-math.atan2(L2*math.sin(q2), L1+L2*math.cos(q2))
    return np.array([q1,q2])

# 順運動学（FK）：肩と肘の角度から、肘と手先の位置を求めます。
# 画面に腕を描く処理や、手先が容器に届いたかの確認で使います。
def fk(q):
    elbow = BASE+L1*np.array([math.cos(q[0]), math.sin(q[0])])
    end = elbow+L2*np.array([math.cos(q.sum()), math.sin(q.sum())])
    return elbow, end

# 今の関節角aから目標bまでを、進み具合t（0〜1）に応じて補間します。
def interpolate(a,b,t):
    # smoothstepという式で、動き始めと止まるときを緩やかにします。
    t = np.clip(t,0,1); t = t*t*(3-2*t)
    # 角度の折り返しを考慮し、近い回転方向へ動かします。
    delta = (b-a+np.pi)%(2*np.pi)-np.pi
    return a+delta*t
