"""Checks model on unseen images, IK round trips and camera-based localization."""
# このファイル：AIとアームの計算が正しく動くか、自動テストで確認します。
import unittest
import numpy as np
from vision import Classifier, features, synthetic, detect, bottle, COLORS, train, ROOT
from robot import ik, fk, BINS, HOME

# 各test_関数が1つの確認項目です。条件に合わないとテストが失敗します。
class CoreChecks(unittest.TestCase):
    # テストの準備：学習済みモデルがなければ作り、共通で読み込みます。
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'model.npz').exists(): train()
        cls.model=Classifier()

    # 学習とは別の乱数シードで300枚を作り、98%以上正解することを確認します。
    def test_unseen_images(self):
        rng=np.random.default_rng(2026)
        correct=0
        for label in range(3):
            for _ in range(100):
                p=self.model.predict(features(synthetic(rng,label)))[0]
                correct+=int(p.argmax()==label)
        self.assertGreaterEqual(correct/300,.98)

    # 既知の位置に3種類を描き、種類・中心位置・信頼度を確認します。
    def test_localization_and_class(self):
        image=np.full((480,800,3),240,np.uint8)
        for i,c in enumerate(COLORS): bottle(image,100+100*i,150,c)
        detections=detect(image,self.model)
        self.assertEqual([d['label'] for d in detections],['A','B','C'])
        for i,d in enumerate(detections):
            self.assertLess(abs(d['x']-(100+100*i)),2)
            self.assertLess(abs(d['y']-150),2)
            self.assertGreater(d['confidence'],.85)

    # 容器のない画像から、存在しない容器を検出しないことを確認します。
    def test_empty_frame(self):
        self.assertEqual(detect(np.full((480,800,3),240,np.uint8),self.model),[])

    # 目標位置→関節角→手先位置と往復計算し、元の位置に戻ることを確認します。
    def test_kinematics(self):
        targets=[HOME,*BINS.values()]+[np.array([x,y]) for x in (110,210,310) for y in (105,197,289)]
        for p in targets: np.testing.assert_allclose(fk(ik(p))[1],p,atol=1e-7)
        with self.assertRaises(ValueError): ik([10000,10000])

# 直接実行したときだけ、テストを開始します。
if __name__=='__main__': unittest.main()
