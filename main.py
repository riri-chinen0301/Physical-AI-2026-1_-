"""ROS 2 camera/vision/controller/simulator demo and reproducible video export.

This is an original 2D kinematic simulator, not Gazebo or a physics engine.
"""
# このファイル：ROS 2でカメラ・AI・アームをつなぎ、シミュレーションと動画を作ります。
# 読む順番は World（世界）→ Vision（認識）→ Controller（動作判断）→ main（実行）です。
import argparse
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
# ROSのメッセージ型です。Imageは画像、JointStateは関節状態、Stringは文字列を運びます。
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from vision import ROOT, COLORS, LABELS, Classifier, bottle, detect, train
from robot import BASE, BINS, HOME, L1, L2, ik, fk, interpolate

# 動画は1秒15枚・90秒です。シミュレーションは全部で1,350ステップ進みます。
FPS = 15
TOTAL_SECONDS = 90
# 日本語テロップを表示するため、Macにある日本語フォントを探します。
FONT_PATH = next(iter(Path('/System/Library/Fonts').glob('*角*W3.ttc')), None)
if FONT_PATH is None:
    FONT_PATH = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
FONTS = {s:ImageFont.truetype(str(FONT_PATH), s) for s in [14,16,18,20,24,28,36,42]}

# 辞書や配列をJSONという文字列形式にして、ROSのStringメッセージへ入れます。
def msg_json(data): return String(data=json.dumps(data))

# 世界を管理するROSノード：容器の配置、仮想カメラ、把持・解放の結果を扱います。
class World(Node):
    def __init__(self, seed):
        super().__init__('medicine_camera_simulator')
        rng = np.random.default_rng(seed)
        # A/B/Cを3個ずつ用意し、投入トレー内の種類の並びだけをランダムにします。
        labels = rng.permutation([0,0,0,1,1,1,2,2,2])
        # 各容器の正解ラベルは評価用です。AIへは画像だけを送り、このラベルは渡しません。
        self.objects = []
        for i, label in enumerate(labels):
            self.objects.append(dict(id=i, label=LABELS[label], x=110.+(i%3)*100,
                                     y=105.+(i//3)*92, status='tray',
                                     color=np.clip(COLORS[label]+rng.normal(0,7,3),0,255).astype(int).tolist()))
        # qは関節角、closedはグリッパーの開閉、attachedは現在つかんでいる容器です。
        self.q = ik(HOME); self.closed = False; self.attached = None
        self.results = []; self.images = 0; self.commands = 0
        # publisherは情報を送る窓口、subscriptionは情報を受け取る窓口です。
        # /で始まる名前はトピック（情報の通り道）です。
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 2)
        self.sub = self.create_subscription(String, '/arm/command', self.command, 10)
        self.joints = self.create_publisher(JointState, '/joint_states', 10)
        self.feedback = self.create_publisher(String, '/arm/feedback', 10)
        self.phase = '待機'; self.last_feedback = None

    # アームへの指令を受け取るたびに、関節角と容器の状態を更新します。
    def command(self, message):
        state = json.loads(message.data); self.commands += 1
        self.q = np.array(state['q']); self.closed = state['closed']; self.phase = state['phase']
        # 指令された角度から手先の位置を計算します。
        _, end = fk(self.q)
        # 手先が容器の中心から9ピクセル以内で閉じたら、把持成功とします。
        if self.closed and self.attached is None:
            candidates = [o for o in self.objects if o['status']=='tray' and np.linalg.norm(np.array([o['x'],o['y']])-end)<9]
            if candidates:
                self.attached = candidates[0]; self.attached['status'] = 'held'
                self.feedback.publish(msg_json(dict(event='grasped', id=self.attached['id'])))
        # つかんだ容器は手先と一緒に移動します。グリッパーを開いたら解放します。
        if self.attached is not None:
            self.attached['x'],self.attached['y'] = map(float,end)
            if not self.closed:
                item = self.attached
                # 解放した場所に一番近い箱を調べ、箱の中心から55ピクセル以内か確認します。
                nearest = min(BINS, key=lambda k:np.linalg.norm(end-BINS[k]))
                distance = float(np.linalg.norm(end-BINS[nearest]))
                in_bin = distance < 55
                item['status'] = 'sorted' if in_bin else 'dropped'
                # 実際に置いた箱と正解を照合します。この正解情報は結果確認にだけ使います。
                result = dict(id=item['id'], truth=item['label'], destination=nearest if in_bin else None,
                              correct=in_bin and item['label']==nearest)
                self.results.append(result)
                self.feedback.publish(msg_json(dict(event='released', **result)))
                self.attached = None
        # 肩・肘・グリッパーの状態をROSへ配信し、外部からも状態を参照できるようにします。
        joint = JointState()
        joint.header.stamp = self.get_clock().now().to_msg()
        joint.name = ['shoulder','elbow','gripper']; joint.position = [*map(float,self.q),float(self.closed)]
        self.joints.publish(joint)

    # 仮想上面カメラの画像を作り、ROSのImageメッセージとして送ります。
    def camera(self):
        image = np.full((480,800,3), (238,242,246), np.uint8)
        cv2.rectangle(image,(45,50),(365,335),(251,252,253),-1)
        for item in self.objects:
            if item['status']=='tray': bottle(image,item['x'],item['y'],item['color'])
        # 簡略化のため、カメラには投入トレー上の容器だけを描きます。アームの遮蔽は省略します。
        # Virtual overhead camera is cropped to the input tray: robot/bins are outside its layer.
        # RGB画像を1本のバイト列へ変換します。stepは画像1行あたりのバイト数です。
        message = Image(); message.height=480; message.width=800; message.encoding='rgb8'
        message.step=800*3; message.data=image.tobytes(); message.header.frame_id='overhead_camera'
        message.header.stamp=self.get_clock().now().to_msg()
        self.publisher.publish(message); self.images += 1
        return image

# 認識を担当するROSノード：カメラ画像を受け取り、AIの判定結果を送ります。
class Vision(Node):
    def __init__(self, model):
        super().__init__('medicine_ai_vision')
        self.model=model; self.frames=0; self.latest=[]
        self.publisher=self.create_publisher(String,'/vision/detections',10)
        self.sub=self.create_subscription(Image,'/camera/image_raw',self.receive,2)

    def receive(self,message):
        # 受け取ったバイト列を、縦×横×RGBの画像配列に戻します。
        image=np.frombuffer(message.data,dtype=np.uint8).reshape(message.height,message.width,3)
        # 容器の中心位置・A/B/Cの判定・信頼度を求め、制御ノードへ送ります。
        self.latest=detect(image,self.model); self.frames+=1
        self.publisher.publish(msg_json(self.latest))

# 動作を判断するROSノード：認識結果から対象を選び、把持→運搬→解放を進めます。
class Controller(Node):
    def __init__(self):
        super().__init__('medicine_arm_controller')
        self.sub=self.create_subscription(String,'/vision/detections',self.receive,10)
        self.feedback_sub=self.create_subscription(String,'/arm/feedback',self.feedback,10)
        self.publisher=self.create_publisher(String,'/arm/command',10)
        self.latest=[]; self.target=None; self.start=0.; self.q=ik(HOME)
        self.origin=self.q.copy(); self.completed=0; self.choices=[]; self.phase='待機'
        self.last_feedback=None; self.error=None; self.feedback_count=0

    def receive(self,message): self.latest=json.loads(message.data)
    def feedback(self,message):
        self.last_feedback=json.loads(message.data); self.feedback_count+=1

    # シミュレーション時刻tに応じて、今の工程に必要なアーム指令を1回分作ります。
    def step(self,t):
        closed=False
        # 最初の6秒は説明表示です。作業中でなければ、信頼度85%以上の候補を選びます。
        if self.target is None and t>=6 and self.completed<9:
            options=[o for o in self.latest if o['confidence']>=.85]
            if options:
                self.target=options[0]; self.start=t; self.origin=self.q.copy()
                self.choices.append(dict(time=t,**self.target)); self.last_feedback=None
        if self.target is not None:
            # 作業開始からの経過時間と、把持位置・箱の中心に届く関節角を求めます。
            age=t-self.start; pick=ik([self.target['x'],self.target['y']])
            place=ik(BINS[self.target['label']])
            # 0〜2秒：容器へ移動。2〜3秒：閉じる。3〜5秒：箱へ運ぶ。
            # 5〜6秒：開く。6〜8秒：待機位置へ戻る、という工程を繰り返します。
            if age<2:
                self.phase='把持位置へ移動'; self.q=interpolate(self.origin,pick,age/2)
            elif age<3:
                self.phase='グリッパーを閉じる'; self.q=pick; closed=True
            elif age<5:
                self.phase='分類先へ運搬'; self.q=interpolate(pick,place,(age-3)/2); closed=True
            elif age<6:
                self.phase='容器を放す'; self.q=place
            elif age<8:
                self.phase='待機位置へ戻る'; self.q=interpolate(place,ik(HOME),(age-6)/2)
            else:
                # 予定時刻になっただけでは成功にしません。解放先が正しいという応答を確認します。
                if not self.last_feedback or self.last_feedback.get('event')!='released' or not self.last_feedback.get('correct'):
                    raise RuntimeError(f'Grasp/release verification failed: {self.last_feedback}')
                self.completed+=1; self.target=None; self.phase='次の容器を確認'
        elif t<6: self.phase='カメラ画像から認識中'
        elif self.completed==9: self.phase='仕分け完了'
        # 関節角・グリッパー開閉・工程名をシミュレータへ送ります。
        self.publisher.publish(msg_json(dict(q=self.q.tolist(),closed=closed,phase=self.phase)))

# 説明・作業台・アーム・カメラ画像・認識結果を、1枚の動画フレームに描きます。
def dashboard(world,vision,controller,camera,t,metrics):
    canvas=PILImage.new('RGB',(1280,720),'#0c1424'); d=ImageDraw.Draw(canvas)
    def text(x,y,s,size=18,color='#dce6f5'): d.text((x,y),s,font=FONTS[size],fill=color)
    def panel(box,fill='#152238',outline='#2a3c55'): d.rounded_rectangle(box,16,fill,outline,width=1)
    text(34,20,'AI 薬剤仕分けロボット',36,'#ffffff')
    text(36,71,'ROS 2 Jazzy  /  学習済みMLP  /  2関節アームの2D運動学シミュレーション',16,'#9eafc7')
    d.rounded_rectangle((1045,26,1245,62),16,'#123c38'); text(1061,31,'模擬容器・実薬不使用',16,'#77ead1')
    panel((24,111,858,635)); panel((882,111,1256,635))
    text(43,125,'上面図：認識 → 把持 → 運搬 → 解放',20)
    # シミュレーション内の座標を、画面上の作業台の位置へずらすための値です。
    ox,oy=40,153
    d.rounded_rectangle((ox+40,oy+45,ox+365,oy+335),12,'#edf2f7')
    text(ox+52,oy+52,'投入トレー',16,'#65758d')
    # 仕分け箱と、それぞれに入った容器の数を描きます。
    for label,point in BINS.items():
        index=LABELS.index(label); color=tuple(map(int,COLORS[index]))
        x,y=point+np.array([ox,oy]); count=sum(r['destination']==label for r in world.results)
        d.rounded_rectangle((x-60,y-51,x+60,y+51),12,fill='#23334a',outline=color,width=3)
        text(x-49,y-43,f'{label}  BOX',16,color)
        text(x-49,y+22,f'{count} / 3',16,'#dce6f5')
    # 容器を描きます。つかんでいる容器は、この後に手先の位置へ描きます。
    for item in world.objects:
        if item['status']=='held': continue
        x,y=item['x']+ox,item['y']+oy
        if item['status']=='sorted':
            k=sum(r['destination']==item['label'] and r['id']<item['id'] for r in world.results)
            center=BINS[item['label']]; x=ox+center[0]-25+k*25; y=oy+center[1]+2
            radius=10
        else: radius=18
        color=tuple(item['color'])
        d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color,outline='#41516b',width=2)
        d.line((x-6,y-4,x+6,y-4),fill='white',width=3)
    # AIが選んだ容器には緑の枠と分類結果を表示します。
    if controller.target:
        target=controller.target; x,y=target['x']+ox,target['y']+oy
        d.rectangle((x-25,y-25,x+25,y+25),outline='#18b99a',width=2)
        text(x-25,y-48,f"{target['label']} {target['confidence']:.0%}",14,'#126d60')
    # 順運動学で求めた位置を使い、肩・肘・手先を線で結んで腕を描きます。
    elbow,end=fk(world.q)
    base=tuple(BASE+np.array([ox,oy])); el=tuple(elbow+[ox,oy]); tip=tuple(end+[ox,oy])
    d.line((base,el,tip),fill='#080d17',width=25)
    d.line((base,el),fill='#b7c8e1',width=17); d.line((el,tip),fill='#6f9cce',width=14)
    for p,r in [(base,24),(el,13),(tip,9)]:
        d.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill='#d5e3f4',outline='#3a5372',width=3)
    gap=8 if world.closed else 17
    d.line((tip[0]-gap,tip[1]-8,tip[0]-gap,tip[1]+9),fill='#48dfc1',width=4)
    d.line((tip[0]+gap,tip[1]-8,tip[0]+gap,tip[1]+9),fill='#48dfc1',width=4)
    if world.attached:
        x,y=tip; d.ellipse((x-14,y-14,x+14,y+14),fill=tuple(world.attached['color']))
    text(904,128,'AI CAMERA',20,'#7ce5d0')
    # 右側には、AIが実際に受け取っている仮想カメラ画像を表示します。
    thumb=PILImage.fromarray(camera[:350,:400]).resize((320,200))
    canvas.paste(thumb,(907,169))
    text(905,378,'モデル：6 → 20 → 3 / MLP',18)
    text(905,410,f"合成画像のテスト精度  {metrics['accuracy']:.1%}",16,'#98aac5')
    target=controller.target
    text(905,451,f"認識結果    {target['label'] if target else '—'}",24,'#ffffff')
    text(905,491,f"信頼度      {target['confidence']:.1%}" if target else '信頼度      —',18)
    text(905,534,'処理状態',14,'#98aac5'); text(905,562,controller.phase,20,'#7ce5d0')
    done=len(world.results); correct=sum(r['correct'] for r in world.results)
    text(35,650,f'仕分け {done} / 9     正しい箱 {correct} / {done}     カメラ受信 {vision.frames}',20)
    text(35,686,'実薬の識別性能を示すものではありません。色で区別した模擬容器を使用。',14,'#91a3bd')
    text(996,653,f'SIM  {t:05.1f} / 90.0 s',18,'#7ce5d0')
    text(978,685,'時間軸1倍・オフライン書き出し',14,'#91a3bd')
    # 冒頭6秒は作品の目的を表示します。
    if t<6:
        d.rounded_rectangle((135,280,750,440),20,fill='#0d2037',outline='#5aa8bd',width=2)
        text(160,300,'カメラ画像をAIで分類',28,'white')
        text(160,351,'A・B・Cの容器を、対応する箱へ運びます。',20)
        text(160,393,'画像の色統計 → ニューラルネット → アーム制御',16,'#7ce5d0')
    # 9個すべての作業を確認できたら、完了メッセージを表示します。
    if controller.completed==9:
        d.rounded_rectangle((140,290,752,433),20,fill='#0d3030',outline='#5de1bd',width=2)
        text(168,308,f'仕分け完了  {correct} / 9',36,'#94f5d9')
        text(168,369,'認識結果と把持・解放のフィードバックを確認',18,'#dce6f5')
    return np.asarray(canvas)

# プログラムの入口です。設定を読み、AI学習・ROS起動・動画出力を順番に行います。
def main():
    # 実行オプション：--seedは容器の種類配置、--previewは画面表示です。
    # --realtimeは実時間に合わせ、--no-videoは動画を作らず動作だけ確認します。
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--realtime',action='store_true'); parser.add_argument('--preview',action='store_true')
    parser.add_argument('--no-video',action='store_true'); args=parser.parse_args()
    # 起動のたびにAIを学習し直し、評価結果を保存します。
    metrics=train()
    os.environ.setdefault('ROS_LOG_DIR',str(ROOT/'logs'))
    Path(os.environ['ROS_LOG_DIR']).mkdir(exist_ok=True)
    # ROSを初期化します。Executorは、各ノードに届いたメッセージの処理を順番に呼びます。
    rclpy.init(); executor=SingleThreadedExecutor()
    world=World(args.seed); vision=Vision(Classifier()); controller=Controller()
    for node in (world,vision,controller): executor.add_node(node)
    writer=None; started=time.monotonic(); snapshots=set()
    # 動画を書き出す場合は、H.264形式のMP4保存先を開きます。
    if not args.no_video:
        writer=cv2.VideoWriter(str(ROOT/'demo.mp4'),cv2.VideoWriter_fourcc(*'avc1'),FPS,(1280,720))
        if not writer.isOpened():
            raise RuntimeError('H.264 video encoder unavailable')
    try:
        # 最初にROS同士が接続され、画像と認識結果が届いたことを確認してから進みます。
        # Discovery handshake: require real camera and detections messages before advancing simulation.
        deadline=time.monotonic()+30
        while (vision.frames<2 or not controller.latest) and time.monotonic()<deadline:
            world.camera()
            for _ in range(12): executor.spin_once(timeout_sec=.01)
        if not controller.latest: raise RuntimeError('ROS 2 camera-to-vision-to-controller pipeline timed out')
        # 1フレームずつ「撮影→認識→指令→移動→描画」を繰り返します。
        for frame in range(TOTAL_SECONDS*FPS):
            t=frame/FPS
            camera=world.camera()
            previous=vision.frames
            deadline=time.monotonic()+2
            # カメラ画像が実際に認識ノードへ届くまで、最大2秒待ちます。
            while vision.frames<=previous and time.monotonic()<deadline: executor.spin_once(timeout_sec=.001)
            if vision.frames<=previous: raise RuntimeError('Camera message lost / vision stalled')
            # 認識ノードが送った検出結果を、制御ノードにも受信させます。
            # Dispatch the detections that were just published.
            for _ in range(3): executor.spin_once(timeout_sec=.001)
            # 制御指令が世界ノードへ届いたか、受信数が増えることで確認します。
            before=world.commands; controller.step(t)
            deadline=time.monotonic()+2
            while world.commands<=before and time.monotonic()<deadline: executor.spin_once(timeout_sec=.001)
            if world.commands<=before: raise RuntimeError('Arm command not delivered')
            for _ in range(2): executor.spin_once(timeout_sec=.001)
            picture=dashboard(world,vision,controller,camera,t,metrics) if writer or args.preview else None
            if writer: writer.write(cv2.cvtColor(picture,cv2.COLOR_RGB2BGR))
            # 動画の代表的な場面をPNG画像としても保存します。
            if writer and frame in (0,225,675,1250):
                cv2.imwrite(str(ROOT/f'preview_{frame:04}.png'),cv2.cvtColor(picture,cv2.COLOR_RGB2BGR))
            if args.preview:
                cv2.imshow('AI Medicine Sorter',cv2.cvtColor(picture,cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1)==27: raise KeyboardInterrupt
            if frame%150==0: print(f'SIM {t:.0f}s: sorted={len(world.results)}/9, vision={vision.frames}',flush=True)
            # 実時間表示では、次のフレームの予定時刻まで待ちます。処理が遅い場合は待ちません。
            if args.realtime:
                wait=started+(frame+1)/FPS-time.monotonic()
                if wait>0: time.sleep(wait)
        # 仕分け数・正解数・AIの判断履歴・ROS受信数を、確認用ファイルへまとめます。
        results=dict(seed=args.seed, simulation_seconds=TOTAL_SECONDS, fps=FPS,
                     camera_frames_received=vision.frames, commands_received=world.commands,
                     feedback_messages=controller.feedback_count, sorted=len(world.results),
                     correct=sum(r['correct'] for r in world.results), completed=controller.completed,
                     objects=world.results, decisions=controller.choices, training=metrics)
        result_path=ROOT/(f'run_results_seed_{args.seed}.json' if args.no_video else 'run_results.json')
        result_path.write_text(json.dumps(results,indent=2))
        # 9個すべての成功を確認できない場合は、成功メッセージを出さずエラーにします。
        if results['correct']!=9 or results['completed']!=9: raise RuntimeError('Not all containers sorted correctly')
        print('PASS: 9/9 containers sorted through ROS 2 image/inference/control pipeline',flush=True)
    # 成功・失敗・中断のどの場合でも、動画ファイルとROSを後片付けします。
    finally:
        if writer: writer.release()
        cv2.destroyAllWindows()
        executor.shutdown()
        for node in (world,vision,controller): node.destroy_node()
        rclpy.shutdown()

# このファイルを直接実行したときに、上のmain関数を呼び出します。
if __name__=='__main__': main()
