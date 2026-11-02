"""Synthetic images -> learned MLP classifier. No class thresholds at inference."""
# このファイル：練習用の画像を作り、AIを学習させ、カメラ画像の容器を分類します。
# NumPyは数値計算、OpenCV（cv2）は画像の作成・加工に使います。
from pathlib import Path
import json
import cv2
import numpy as np

# 保存先は、このPythonファイルがあるフォルダです。
ROOT = Path(__file__).resolve().parent
# 赤・青・黄を、教材用の容器A・B・Cの見た目として使います。
COLORS = np.array([[218, 62, 81], [53, 126, 225], [232, 186, 45]])
LABELS = ['A', 'B', 'C']

# 容器を上から見た図を描きます。円がふた、白い線が光の反射です。
# xとyは画像内の中心位置、radiusは円の半径です。
def bottle(image, x, y, color, radius=18):
    x, y, radius = int(x), int(y), int(radius)
    cv2.circle(image, (x+3, y+4), radius+2, (208, 211, 215), -1)
    cv2.circle(image, (x, y), radius, tuple(int(c) for c in color), -1)
    cv2.circle(image, (x, y), radius, (55, 65, 80), 2)
    cv2.line(image, (x-7, y-5), (x+7, y-5), (245, 247, 251), 3)
    cv2.circle(image, (x-6, y-9), 3, (245, 247, 251), -1)

# 背景と容器を分けます。HSVは色相・鮮やかさ・明るさで色を表す方法です。
# ここでのしきい値は「容器の場所」を探すためのものです。A/B/Cの種類はAIが選びます。
def foreground(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return ((hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 45)).astype(np.uint8)

# 画像を、その特徴を表す6個の数字へ変換します。
# RGBそれぞれの平均3個＋ばらつき（標準偏差）3個をAIへ渡します。
def features(image):
    # 背景を除いた画素だけを使い、0〜255の色を0〜1へそろえます。
    pixels = image[foreground(image).astype(bool)] / 255.
    if len(pixels) < 15:
        raise ValueError('Insufficient foreground pixels')
    return np.r_[pixels.mean(axis=0), pixels.std(axis=0)]

# 学習用の合成画像を1枚作ります。正解ラベルは画像生成時だけ指定します。
# 同じ種類でも明るさ・位置・大きさ・ノイズを変え、単なる画像の丸暗記を防ぎます。
def synthetic(rng, label):
    image = np.full((64, 64, 3), rng.integers(220, 251), dtype=np.uint8)
    color = np.clip(COLORS[label] * rng.uniform(.72, 1.12) + rng.normal(0, 11, 3), 0, 255)
    bottle(image, rng.integers(26, 39), rng.integers(26, 39), color, rng.integers(13, 21))
    noise = rng.normal(0, 2.5, image.shape)
    return np.clip(image.astype(float)+noise, 0, 255).astype(np.uint8)

# 3種類のスコアを、合計が1になる数値へ変換します。
# この値を画面では信頼度として表示しますが、正しさの保証ではありません。
def softmax(x):
    e = np.exp(x-x.max(axis=1, keepdims=True))
    return e/e.sum(axis=1, keepdims=True)

# 学習結果を読み込み、特徴の数字からA/B/Cのスコアを計算するクラスです。
class Classifier:
    # model.npzには、学習した重みと入力をそろえるための平均・標準偏差が入っています。
    def __init__(self, path=None):
        self.params = dict(np.load(path or ROOT/'model.npz'))

    # 推論＝学習済みAIで新しい入力の答えを予測する処理です。
    def predict(self, feature):
        p = self.params
        # 学習時と同じ基準で特徴を正規化してから、隠れ層と出力層を計算します。
        # @ は行列の掛け算、tanhは値を非線形に変換する関数です。
        x = (np.atleast_2d(feature)-p['mean'])/p['std']
        return softmax(np.tanh(x@p['w1']+p['b1'])@p['w2']+p['b2'])

# 画像の生成→学習→未使用画像での評価→結果保存をまとめて実行します。
def train():
    # 乱数シードを固定し、毎回同じ学習データと結果を再現できるようにします。
    rng = np.random.default_rng(42)
    x, y = [], []
    montage = np.full((3*64, 10*64, 3), 255, np.uint8)
    # 各種類500枚ずつ、合計1,500枚の画像を作ります。
    for label in range(3):
        for i in range(500):
            image = synthetic(rng, label)
            x.append(features(image)); y.append(label)
            if i < 10:
                montage[label*64:(label+1)*64, i*64:(i+1)*64] = image
    x, y = np.array(x), np.array(y)
    # データを混ぜ、学習用1,200枚とテスト用300枚に分けます。
    order = rng.permutation(len(y)); train_ids, test_ids = order[:1200], order[1200:]
    # 正規化の基準は学習用だけから求め、テスト画像の情報を学習に混ぜません。
    mean, std = x[train_ids].mean(0), x[train_ids].std(0)+1e-5
    z = (x[train_ids]-mean)/std
    # 正解を、A=[1,0,0]・B=[0,1,0]・C=[0,0,1]の形に変えます。
    targets = np.eye(3)[y[train_ids]]
    # 重みは、AIが学習によって調整する数字です。入力6→隠れ層20→出力3の構造です。
    w1 = rng.normal(0, .2, (6, 20)); b1 = np.zeros(20)
    w2 = rng.normal(0, .2, (20, 3)); b2 = np.zeros(3)
    history = []
    # 予測と正解の差が小さくなるように、重みの更新を700回繰り返します。
    for epoch in range(700):
        h = np.tanh(z@w1+b1); prob = softmax(h@w2+b2)
        # 誤差逆伝播：出力側の誤差から、各重みをどちらへ変えるべきか計算します。
        delta = (prob-targets)/len(z)
        dh = (delta@w2.T)*(1-h*h)
        dw1, db1 = z.T@dh, dh.sum(0)
        dw2, db2 = h.T@delta, delta.sum(0)
        for param, grad in [(w1,dw1), (b1,db1), (w2,dw2), (b2,db2)]:
            # 学習率0.12で、誤差を減らす方向へ少しずつ数字を更新します。
            param -= .12*grad
        if epoch % 100 == 0:
            history.append(float(-np.log(prob[np.arange(len(z)), y[train_ids]]+1e-12).mean()))
    # 学習した数字を保存します。次の推論では、このファイルを読み込めます。
    np.savez(ROOT/'model.npz', mean=mean, std=std, w1=w1, b1=b1, w2=w2, b2=b2)
    # 学習に使っていない300枚で評価します。argmaxは最大スコアの種類を選びます。
    pred = Classifier().predict(x[test_ids]).argmax(1)
    # 混同行列は、正解の種類ごとに何を予測したかをまとめた表です。
    confusion = np.zeros((3,3), int)
    for a,b in zip(y[test_ids], pred): confusion[a,b] += 1
    result = dict(train_samples=1200, test_samples=300, accuracy=float((pred==y[test_ids]).mean()),
                  confusion_matrix=confusion.tolist(), loss=history, seed=42,
                  model='6 image color statistics -> 20 tanh -> 3 softmax',
                  limitation='Synthetic colors only; no validation on real medicines or real camera images.')
    (ROOT/'training_metrics.json').write_text(json.dumps(result, indent=2))
    cv2.imwrite(str(ROOT/'training_examples.png'), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return result

# カメラ画像から「位置・大きさ・AIが選んだ種類・信頼度」の一覧を返します。
def detect(image, model):
    # 前景抽出で場所を探し、学習したネットワークでA/B/Cを判定します。
    # Segmentation locates colored objects; the learned network assigns A/B/C.
    mask = foreground(image)
    # 右側の箱と下側のアーム領域を対象外とし、投入トレーだけを調べます。
    mask[:, 400:] = 0
    mask[350:, :] = 0
    # 輪郭＝画像の中でつながっている前景の外周です。容器ごとに切り出します。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        # 小さい点やノイズは容器として扱いません。
        if cv2.contourArea(contour) < 180: continue
        x,y,w,h = cv2.boundingRect(contour)
        # 切り出した画像の特徴をAIへ渡します。正解ラベルは渡しません。
        prob = model.predict(features(image[y:y+h, x:x+w]))[0]
        results.append(dict(x=x+w/2, y=y+h/2, box=[x,y,w,h],
                            label=LABELS[int(prob.argmax())], confidence=float(prob.max())))
    # 見つけた候補を、おおむね上の列から、各列では左から扱う順に並べます。
    return sorted(results, key=lambda o: (round(o['y']/60), o['x']))

# このファイルを直接実行した場合だけ、学習と評価を行います。
if __name__ == '__main__':
    print(json.dumps(train(), indent=2))
