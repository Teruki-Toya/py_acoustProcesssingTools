# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:27:03 2023

@author: Teruki Toya
"""
import flet as ft
import soundfile as sf
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart
import scipy.fftpack as spfft
from scipy import signal

# 固定パラメータの定義 ---------------
#sd.default.device = [1, 4] # Input, Outputデバイスを指定
input_device_info = sd.query_devices(device=sd.default.device[1])
FSu = 44100 # サンプリングレート [samples/sec]

T_rec = 5.0
T = 2.0         # 信号長 [sec]
N_rec = round(FSu*T_rec)
N = round(FSu*T) # 信号のサンプル数 [samples]
winSize_fine = 128  # スペクトログラム描画時のフレーム長
shiftSize_fine = int(winSize_fine / 4)  # シフト長
winSize_env = 64    # フォルマント抽出時のフレーム長
shiftSize_env = int(winSize_env / 4)  # シフト長
fs = 8000     # フォルマント抽出時のサンプリングレート [samples/sec]
f0L = 80      # 基本周波数の下限 [Hz]
f0U = 250     # 基本周波数の上限 [Hz]
f_pass = 4000   # LPF通過域 [Hz]
f_stop = 6000   # LPF阻止域 [Hz]
g_pass = 3      # LPF通過域最大損失 [dB]
g_stop = 40     # LPF阻止域最大損失 [dB]
lpcOrder = 10   # 線形予測次数
N_peaks = int(lpcOrder / 2)  # 声道フィルタで想定するピーク数

########## ここから関数群 ##########

## ハニング窓
def winHann(winSize):
    n = np.linspace(0, 1, winSize)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n)
    return w

## FFTで振幅スぺクトルを計算する関数
def FFTspectrum(x, fs, fftSize):
  
    # 高速フーリエ変換（FFT）
    X = spfft.fft(x, fftSize) / fftSize
  
    # 分析対象の周波数列
    idx = np.linspace(0, int(fftSize/2), int(fftSize/2+1))
    freq = idx * fs / fftSize
  
    # 振幅スペクトル
    S_amp = 20 * np.log10(abs(X))
    S_amp = S_amp[0 : len(freq)]
  
    return freq, S_amp

## 自己相関の計算
def corrAuto(x, order = 0):
    if order <= 0:
        r = np.zeros(len(x))
    else:
        r = np.zeros(order + 1)

    for m in range(len(r)):
        for n in range(len(x) - m):
            r[m] = r[m] + x[n] * x[n + m]
  
    return r

## 自己相関法による基本周波数軌跡の検出
def F0contour(x, fs, f_lower, f_upper):
    # 自己相関
    r = corrAuto(x)
    
    I_maxPeak = np.zeros(4)
    I_maxPeak = I_maxPeak.astype(int)
    ipk, _ = signal.find_peaks(r)  # 自己相関のピークを検出
    Pks = r[ipk]
    
    # 自己相関に現れるピークの位置を検出してはひたすら差分する
    if len(Pks) > 0:
        i = np.arange(len(Pks), dtype = int)
        I_maxPeak[0] = i[Pks == np.amax(Pks)]
        if len(Pks) > I_maxPeak[0]:
            Pks1 = Pks[I_maxPeak[0] : len(Pks)]
            i1 = np.arange(len(Pks1), dtype = int)
            I_maxPeak[1] = i1[Pks1 == np.amax(Pks1)]
            if len(Pks1) > I_maxPeak[1]:
                Pks2 = Pks1[I_maxPeak[1] : len(Pks1)]
                i2 = np.arange(len(Pks2), dtype = int)
                I_maxPeak[2] = i2[Pks2 == np.amax(Pks2)]
                if len(Pks2) > I_maxPeak[2]:
                    Pks3 = Pks2[I_maxPeak[2] : len(Pks2)]
                    i3 = np.arange(len(Pks3), dtype = int)
                    I_maxPeak[3] = i3[Pks3 == np.amax(Pks3)]
                    
        if I_maxPeak[1]:
            if I_maxPeak[2]:
                if I_maxPeak[3]:
                    d0_ind = [
                        0, ipk[I_maxPeak[0]], ipk[np.sum(I_maxPeak[0:2])],
                        ipk[np.sum(I_maxPeak[0:3])],
                        ipk[np.sum(I_maxPeak[0:4])]
                    ]
                else:
                    d0_ind = [
                        0, ipk[I_maxPeak[0]], ipk[np.sum(I_maxPeak[0:2])],
                        ipk[np.sum(I_maxPeak[0:3])]
                    ]
                    
            else:
                d0_ind = [
                    0, ipk[I_maxPeak[0]], ipk[np.sum(I_maxPeak[0:2])]
                ]
            
            d0_ind = np.diff(d0_ind)
            
        else:
            d0_ind = ipk[I_maxPeak[0]]
        
        f0_candidate = (d0_ind / fs) ** (-1)
        
    else:
        f0_candidate = -1
    
    if f0_candidate >= f_lower and f0_candidate <= f_upper:
        f0_est = f0_candidate
    else:
        f0_est = -1
        
    return f0_est

## Levinson-Durbin 法による線形予測係数（LPC）の計算
def LDcalc(x, lpcOrder):

    # パラメータ（初期値 0 埋め）
    a = np.zeros(lpcOrder + 1)
    lpc = np.zeros(lpcOrder + 1)
    gamma = np.zeros(lpcOrder + 1)
    epsilon = np.zeros(lpcOrder + 1)
  
    # 自己相関
    r = corrAuto(x, lpcOrder)
  
    # パラメータ ε, γ, lpc の設定
    epsilon[0] = r[0]
    gamma[1] = -r[1] / epsilon[0]
    lpc[0] = 1
    lpc[1] = gamma[1]
    epsilon[1] = epsilon[0] * (1 - (gamma[1] ** 2))
  
    # lpc と parcor 係数の計算
    ix = np.arange(lpcOrder + 1)
    for m in ix[2 : len(ix)]:
        for n in range(m):
            a[n] = lpc[n]
	
        a[m] = 0
        num = 0
	
        for n in range(m + 1):
            num = num + a[n] * r[m - n]
	
        gamma[m] = -num / epsilon[m - 1]
	
        for n in range(m + 1):
            lpc[n] = a[n] + gamma[m] * a[m - n]
	
        epsilon[m] = epsilon[m - 1] * (1 - (gamma[m] ** 2))
  
    parcor = -gamma
  
    return lpc, parcor

## 線形予測（LP）によるフォルマント軌跡の検出
def spEnv_lpc_peak(x, fs, lpcOrder):
    fftSize = len(x) * 5
    
    # プリエンファシス
    xtmp = np.append(0, x[0 : len(x) - 1])
    xe = x - 0.98 * xtmp
    xe[0] = 0
  
    # 分析窓をかける
    w = winHann(len(x))
    xew = xe * w
  
    # 線形予測分析
    x_lpc, _ = LDcalc(xew, lpcOrder)
  
    # 声道フィルタの導出
    fVT, invVTfilt = FFTspectrum(x_lpc, fs, fftSize)
            # fVT: 周波数サンプル列
            # invVTfilt : LPC列をフーリエ変換して声道逆フィルタを得る
    VTfilt = - invVTfilt  # dB 表現では符号反転すれば順フィルタとなる
    VTfilt = VTfilt - np.amax(VTfilt)  # 正規化
    
    I_peaks = np.zeros(N_peaks)
    ipkCand, _ = signal.find_peaks(VTfilt)  # 声道フィルタのピーク候補を検出
    if len(ipkCand) <= N_peaks:  # 想定よりピーク候補が少なければ
        I_peaks[0 : len(ipkCand)] = ipkCand  # I_peaks に左詰め
    else:                        # そうでなければ
        I_peaks = ipkCand[0 : N_peaks]  # 想定されるピーク数分，ピーク候補を採用
    
    I_peaks = I_peaks.astype(int)
    F_peaks = fVT[I_peaks]  # 声道フィルタのピーク周波数
    F_peaks[F_peaks <= 0] = -1  # 候補になり得ないケースは -1 として欄外表示

    return F_peaks

# ローパスフィルタ
def lowpass(x, fs, f_pass, f_stop, g_pass, g_stop):
    w_pass = f_pass / (fs/2)                      #ナイキスト周波数で通過域端周波数を正規化
    w_stop = f_stop / (fs/2)                      #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(w_pass, w_stop, g_pass, g_stop)  # オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y    

########## ここまで関数群 ##########

# Flet の処理 ---------------------
def main(page):
    
    page.title = "声のスペクトログラムを観察しよう"  # タイトル
    # page.vertical_alignment = ft.MainAxisAlignment.CENTER

    page.window_width = 750  # 幅
    page.window_height = 900  # 高さ
    page.window_top = 100  # 位置(TOP)
    page.window_left = 100  # 位置(LEFT)
    page.window_always_on_top = True  # ウィンドウを最前面に固定
    page.window_center()  # ウィンドウをデスクトップの中心に移動
    
    setDemoNum = ft.Ref[ft.TextField]()  # デモ音番号
    setA_slider = ft.Ref[ft.Slider]()   # ゲイン調整スライドバー
    
    fig0, ax0 = plt.subplots(figsize=(13, 6))  # 時間波形の図示
    ax0.set_title("Waveform")
    ax0.set_xlabel("Time [s]", fontsize = 13)
    ax0.set_ylabel("Amplitude", fontsize = 13)
    ax0.set_xlim([0, T])
    ax0.set_ylim([-1.05, 1.05])
    
    chkDetail = ft.Ref[ft.Checkbox()]()       # 詳細分析チェック
    
    fig, ax = plt.subplots(figsize=(13, 6))  # スペクトログラムの図示
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time [s]", fontsize = 13)
    ax.set_ylabel("Frequency [Hz]", fontsize = 13)
    ax.set_xlim([0, T])
    ax.set_ylim([0, 3000])

    # 録音ボタン押下時の動作を記述 -----------------------------------
    def recButton_clicked(e):
        
        # 録音スタート合図音を生成
        alert_t = np.arange(int(FSu/2)) / FSu       # 音のサンプル時間列
        alertSnd = 0.97 * np.sin((2*np.pi*1000) * alert_t)
        N_taper_alert = round(0.1 * FSu)   # テーパー部のサンプル数
        taper_alert = np.sin(np.linspace(0, np.pi/2, N_taper_alert)) # 上昇部
        taper_alert = np.concatenate([
                            taper_alert,
                            np.ones(len(alertSnd) - 2*N_taper_alert),
                            np.flipud(taper_alert)
                      ])   # 下降部は上昇部の時間反転数列
        # 録音スタート合図音にテーパー処理を施す
        alertSnd = (10 ** (-30/20)) * alertSnd * taper_alert
        sd.play(alertSnd, FSu)  # 再生
        sd.wait() # 再生終了待ち
        # マイクから録音する
        x_row_rec = sd.rec(N_rec, samplerate=FSu, channels=1)
        sd.wait() # 録音終了待ち
        # 時間サンプル列と振幅調整
        if setA_slider.current.value is not None:
            x_row_rec = (10**(int(setA_slider.current.value)/20)) * x_row_rec
            # スライドバーが動かされなければ current value は 'NonType' のはず．
            # 動かされると current value は str 型になるため，そのときだけゲイン調整
        x_row = x_row_rec[N_rec - N : N_rec]
        t = np.arange(N) / FSu
        # 録音おわり合図音
        sd.play(alertSnd, FSu)  # 再生
        sd.wait() # 録音終了待ち
        # WAVファイルに書き出し
        sf.write("wav/recsnd.wav", x_row, FSu)
        
        sd.play(x_row, FSu)  # 再生
        
        # 描画 － 時間波形
        ax0.cla()
        ax0.plot(t, 2*x_row)
        ax0.set_xlabel("Time [s]", fontsize = 13)
        ax0.set_ylabel("Amplitude", fontsize = 13)
        ax0.set_xlim([0, T])
        ax0.set_ylim([-1.05, 1.05])
        
        page.update()               # ページを更新
        
    # デモ音読込ボタン押下時の動作を記述 -----------------------------------
    def demo_clicked(e):
        
        # デモ用WAVファイル読み込み・録音済として上書き
        if setDemoNum.current.value and int(setDemoNum.current.value) < 4:
            filePath = 'wav/voiceEx_sensei' + setDemoNum.current.value + '.wav'
        else:
            filePath = 'wav/voiceEx_sensei.wav'
        x_row, FSu = sf.read(filePath)
        if setA_slider.current.value is not None:
            x_row = (10**(int(setA_slider.current.value)/20)) * x_row
            # スライドバーが動かされなければ current value は 'NonType' のはず．
            # 動かされると current value は str 型になるため，そのときだけゲイン調整
            
        sf.write("wav/recsnd.wav", x_row, FSu)
        t = np.arange(N) / FSu
        
        sd.play(x_row, FSu)  # 再生
        
        # 描画 － 時間波形
        ax0.cla()
        ax0.plot(t, 2*x_row)
        ax0.set_xlabel("Time [s]", fontsize = 13)
        ax0.set_ylabel("Amplitude", fontsize = 13)
        ax0.set_xlim([0, T])
        ax0.set_ylim([-1.05, 1.05])
        
        page.update()               # ページを更新
        
        
    # 分析ボタン押下時の動作を記述 -----------------------------------
    def SpgramButton_clicked(e):
        
        # 録音済WAVファイル読み込み
        x_row, FSu = sf.read('wav/recsnd.wav')
        t = np.arange(N) / FSu
        sd.play(x_row, FSu)  # 再生
        x_row_filtered = lowpass(x_row, FSu, f_pass, f_stop, g_pass, g_stop)
        x0 = signal.resample(x_row_filtered, int(fs * T))
                
        # === スペクトログラム描画のためのフレーム処理 =====================
        # 総フレーム数
        N_frame_fine = np.floor(
            (len(x0) - (winSize_fine - shiftSize_fine)) / shiftSize_fine
        )
        N_frame_fine = int(N_frame_fine)
      
        # [フレーム数×フレーム長]のスペクトル行列
        Sg0 = np.zeros((  # 0 埋めした行列を先に作っておく
                        N_frame_fine,
                        int(winSize_fine / 2 + 1)
                     ))
        F0 = -1 * np.ones(N_frame_fine)
        t_s = np.zeros(N_frame_fine) # フレームインデックスに対応する時刻（0埋め）
        for k in range(N_frame_fine):
            offset = shiftSize_fine * k       # フレームをシフトしながら
            x0_tmp = x0[offset : offset + winSize_fine]  # フレーム長分だけ信号を切り出す 

            w = winHann(len(x0_tmp))    # 分析窓をかける
            x_tmp = x0_tmp * w
            
            # 該当フレームに対するFFTスペクトルの出力
            freq, S_tmp = FFTspectrum(x_tmp, fs, winSize_fine)
            
            Sg0[k, :] = S_tmp[::-1]
            t_s[k] = (offset + winSize_fine / 2) / fs
            
            # 該当フレームに対する基本周波数推定
            F0[k] = F0contour(x_tmp, fs, f0L, f0U)
        
        Fqsearch = np.flipud(freq)
        Sg = Sg0.T  # 転置（フレームサンプルを横方向，周波数を縦方向に）
        Sg = Sg - np.amax(Sg)  # 正規化
        
        Sg = Sg[Fqsearch<=4000, :]
        # 描画に必要な変数は t_s（サンプル時刻），freq（周波数），Sg（スペクトル行列）
        
        
        # === フォルマント抽出のためのフレーム処理 =====================
        # 総フレーム数
        N_frame_env = np.floor(
            (len(x0) - (winSize_env - shiftSize_env)) / shiftSize_env
        )
        N_frame_env = int(N_frame_env)
        
        # [ピークの数×フレーム数]のデータ行列
        F_pk = np.zeros((  # 0 埋めした行列を先に作っておく
                        N_peaks,
                        N_frame_env
                     ))
        t_e = np.zeros(N_frame_env) # フレームインデックスに対応する時刻（0埋め）
        for i in range(N_frame_env):
            offset = shiftSize_env * i       # フレームをシフトしながら
            xe_tmp = x0[offset : offset + winSize_env] # フレーム長分だけ信号を切り出す 
            
            # 声道フィルタのピーク周波数を検出
            Fp_tmp = spEnv_lpc_peak(xe_tmp, fs, lpcOrder)
            F_pk[:, i] = Fp_tmp
            t_e[i] = (offset + winSize_env / 2) / fs
        # F_peaksを列（サンプル時刻）ごとに描画していけば，フォルマント軌跡となる
        
        # 時間波形
        # 描画 － 時間波形
        ax0.cla()
        ax0.plot(t, 2*x_row)
        ax0.set_xlabel("Time [s]", fontsize = 13)
        ax0.set_ylabel("Amplitude", fontsize = 13)
        ax0.set_xlim([0, T])
        ax0.set_ylim([-1.05, 1.05])
        
        # スペクトログラム
        ax.cla()
        ax.imshow(Sg, cmap="jet", extent=[0, T, 0, 4000], aspect="auto", vmin=-100, vmax=10)
        ax.set_title("Spectrogram")
        ax.set_xlabel("Time [s]", fontsize = 13)
        ax.set_ylabel("Frequency [Hz]", fontsize = 13)
        ax.set_xlim([0, T])
        ax.set_ylim([0, 3000])
        if chkDetail.current.value == True:  # 詳細チェックあり
            # スペクトログラムの上にフォルマント軌跡を重ねる
            ax.scatter(t_e, F_pk[0, :], marker = "+", color = 'darkviolet')
            ax.scatter(t_e, F_pk[1, :], marker = "x", color = 'black')
            # 時間波形の上に基本周波数（F0）軌跡を重ねる
            ax0_add = ax0.twinx()
            ax0_add.scatter(t_s, F0, marker = "x", color = 'maroon')
            ax0_add.set_ylabel('Frequency [Hz]', color='maroon')
            ax0_add.axis('tight')
            ax0_add.set_ylim([f0L - 10, f0U + 10])
        
        page.update()               # ページを更新

    # Flet コントロールの追加とページへの反映 -------------------------------
    page.add(
        
        ft.Row(
            controls=[
                ft.ElevatedButton(          # 録音ボタン
                    "録音（2 秒）",
                    width = 200,
                    on_click = recButton_clicked
                ),
                ft.Text("　　 "),
                ft.ElevatedButton(          # デモ音読込ボタン
                    "(デモ音を読み込む)",
                    width = 200,
                    on_click = demo_clicked
                ),
                ft.TextField(               # デモ音番号
                    ref  = setDemoNum,
                    label = "No.",
                    width = 75,
                ),
                ft.Text("　ゲイン :"),
                ft.Slider(                  # ゲイン調整スライドバー
                    ref = setA_slider,
                    min = -6,
                    max = 12,
                    divisions = 3,
                    label = "{value} dB",
                    width = 100
                ),
            ]
        ),
        
        
        # 時間波形
        MatplotlibChart(fig0, expand=True),
        
        # 分析区間指定ボックス（開始用）
        ft.Row(
            controls=[
                ft.Text("スペクトログラム"),
                ft.Checkbox(                # 詳細分析チェック
                    ref = chkDetail,
                    label = "フォルマントと基本周波数も分析する"
                )
            ]
        ),
        
        # 分析ボタン
        ft.ElevatedButton("スペクトログラムを出力", on_click = SpgramButton_clicked),
        
        # スペクトログラム
        MatplotlibChart(fig, expand=True),
 
    )

ft.app(target=main)
