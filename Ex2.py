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

# 固定パラメータの定義 ---------------
#sd.default.device = [1, 4] # Input, Outputデバイスを指定
input_device_info = sd.query_devices(device=sd.default.device[1])
fs = int(input_device_info["default_samplerate"]) # サンプリングレート [samples/sec]

T_rec = 5.0
T = 2.0         # 信号長 [sec]
N_rec = round(fs*T_rec)
N = round(fs*T) # 信号のサンプル数 [samples]
winSize = 2048  # FFTフレーム長
liftSize = 96   # リフタカットオフ

########## ここから関数群 ##########

## ハニング窓
def winHann(winSize):
  n = np.linspace(0, 1, winSize)
  w = 0.5 - 0.5 * np.cos(2 * np.pi * n)
  return w

## FFTで振幅スぺクトルを計算する関数
def FFTspectrum_cepstrum(x, fs, fftSize):

  # 分析窓をかける
  w = winHann(len(x))
  xw = x * w
  
  # 高速フーリエ変換（FFT）
  X = spfft.fft(xw, fftSize) / fftSize
  
  # 分析対象の周波数列
  idx = np.linspace(0, int(fftSize/2), int(fftSize/2+1))
  freq = idx * fs / fftSize
  
  # 振幅スペクトル
  S_amp_row = 20 * np.log10(abs(X))
  S_amp = S_amp_row[0 : len(freq)] - np.amax(S_amp_row)
  
  # ケプストラム
  xc = spfft.ifft(S_amp_row, fftSize)
  
  return freq, S_amp, xc

## ケプストラム法によるスペクトル包絡の分析
def envelope_c(xc, liftSize):
   
  # 低ケフレンシ成分以外を 0 埋め
  xc[liftSize : len(xc) - liftSize] = np.zeros(len(xc) - 2 * liftSize)

  # 低ケフレンシ成分のフーリエ変換
  XC = spfft.fft(xc)
  
  # 分析対象の周波数列
  idx = np.linspace(0, int(len(xc)/2), int(len(xc)/2+1))
  freq = idx * fs / len(xc)
  
  # dB 表現（すでに対数スペクトルになっている）
  XC_amp = XC.real
  XC_amp = XC_amp[0 : len(freq)] - np.amax(XC_amp)
  
  return freq, XC_amp

########## ここまで関数群 ##########

# Flet の処理 ---------------------
def main(page):
    
    page.title = "マイクで観測した音を分析"  # タイトル
    # page.vertical_alignment = ft.MainAxisAlignment.CENTER

    page.window_width = 750  # 幅
    page.window_height = 800  # 高さ
    page.window_top = 100  # 位置(TOP)
    page.window_left = 100  # 位置(LEFT)
    page.window_always_on_top = True  # ウィンドウを最前面に固定
    page.window_center()  # ウィンドウをデスクトップの中心に移動
    
    setDemoNum = ft.Ref[ft.TextField]()  # デモ音番号
    setA_slider = ft.Ref[ft.Slider]()   # ゲイン調整スライドバー
    
    fig0, ax0 = plt.subplots()  # 時間波形の図示
    ax0.set_title("Waveform")
    ax0.set_xlabel("Time [s]", fontsize = 13)
    ax0.set_ylabel("Amplitude", fontsize = 13)
    ax0.set_xlim([0, T])
    ax0.set_ylim([-1.05, 1.05])
    
    setWin1Time = ft.Ref[ft.TextField]()  # 分析区間1 時間位置入力ボックス
    setWin2Time = ft.Ref[ft.TextField]()  # 分析区間2 時間位置入力ボックス
    chkEnv = ft.Ref[ft.Checkbox()]()       # スペクトル包絡描画チェック
    
    fig = plt.figure(figsize=(11, 4))  # 図示
    # （左） 分析区間1のスペクトル
    ax1 = fig.add_subplot(121)
    ax1.set_title("Time-1 spectrum")
    ax1.set_xlabel("Frequency [Hz]", fontsize = 13)
    ax1.set_ylabel("Relative level [dB]", fontsize = 13)
    ax1.set_xlim([0, 4000])
    ax1.set_ylim([-55, 5])
    # （右） 分析区間2のスペクトル
    ax2 = fig.add_subplot(122)
    ax2.set_title("Time-2 spectrum")
    ax2.set_xlabel("Frequency [Hz]", fontsize = 13)
    ax2.set_ylabel("Relative level [dB]", fontsize = 13)
    ax2.set_xlim([0, 4000])
    ax2.set_ylim([-55, 5])

    # 録音ボタン押下時の動作を記述 -----------------------------------
    def recButton_clicked(e):
        
        # 録音スタート合図音を生成
        alert_t = np.arange(int(fs/2)) / fs       # 音のサンプル時間列
        alertSnd = 0.97 * np.sin((2*np.pi*1000) * alert_t)
        N_taper_alert = round(0.1 * fs)   # テーパー部のサンプル数
        taper_alert = np.sin(np.linspace(0, np.pi/2, N_taper_alert)) # 上昇部
        taper_alert = np.concatenate([
                            taper_alert,
                            np.ones(len(alertSnd) - 2*N_taper_alert),
                            np.flipud(taper_alert)
                      ])   # 下降部は上昇部の時間反転数列
        # 録音スタート合図音にテーパー処理を施す
        alertSnd = (10 ** (-30/20)) * alertSnd * taper_alert
        sd.play(alertSnd, fs)  # 再生
        sd.wait() # 再生終了待ち
        # マイクから録音する
        x0_rec = sd.rec(N_rec, samplerate=fs, channels=1)
        sd.wait() # 録音終了待ち
        # 時間サンプル列と振幅調整
        if setA_slider.current.value is not None:
            x0_rec = (10**(int(setA_slider.current.value)/20)) * x0_rec
            # スライドバーが動かされなければ current value は 'NonType' のはず．
            # 動かされると current value は str 型になるため，そのときだけゲイン調整
        x0 = x0_rec[N_rec - N : N_rec]
        t = np.arange(N) / fs
        # 録音おわり合図音
        sd.play(alertSnd, fs)  # 再生
        sd.wait() # 再生終了待ち
        # WAVファイルに書き出し
        sf.write("wav/recsnd.wav", x0, fs)
        
        sd.play(x0, fs)  # 再生
        
        # 描画 － 時間波形
        ax0.cla()
        ax0.plot(t, 2*x0)
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
        x0, fs = sf.read(filePath)
        if setA_slider.current.value is not None:
            x0 = (10**(int(setA_slider.current.value)/20)) * x0
            # スライドバーが動かされなければ current value は 'NonType' のはず．
            # 動かされると current value は str 型になるため，そのときだけゲイン調整
            
        sf.write("wav/recsnd.wav", x0, fs)
        t = np.arange(N) / fs
        
        sd.play(x0, fs)  # 再生
        
        # 描画 － 時間波形
        ax0.cla()
        ax0.plot(t, 2*x0)
        ax0.set_xlabel("Time [s]", fontsize = 13)
        ax0.set_ylabel("Amplitude", fontsize = 13)
        ax0.set_xlim([0, T])
        ax0.set_ylim([-1.05, 1.05])
        
        page.update()               # ページを更新
        
        
    # 分析ボタン押下時の動作を記述 -----------------------------------
    def FFTButton_clicked(e):
        chkEnv.disabled = False
        
        # 録音済WAVファイル読み込み
        x0, fs = sf.read('wav/recsnd.wav')
        t = np.arange(N) / fs
        sd.play(x0, fs)  # 再生
        
        # 時間点に対応するサンプルインデックス
        i1_center = round(fs * float(setWin1Time.current.value))
        i2_center = round(fs * float(setWin2Time.current.value))
        
        # 時間点のフレームで信号を切り出し
        i1_start = round(i1_center - winSize / 2)
        i1_end = round(i1_center + winSize / 2)
        y1 = x0[i1_start : i1_end]
        
        i2_start = round(i2_center - winSize / 2)
        i2_end = round(i2_center + winSize / 2)
        y2 = x0[i2_start : i2_end]
        
        # 時間波形上に分析区間を表示
        t1_start = i1_start / fs
        t1_end = i1_end / fs
        t2_start = i2_start / fs
        t2_end = i2_end / fs
        ax0.cla()
        ax0.plot(t, 2*x0)
        ax0.plot([t1_start, t1_start], [-1.05, 1.05], color = 'maroon')
        ax0.plot([t1_end, t1_end], [-1.05, 1.05], color = 'maroon')
        ax0.plot([t2_start, t2_start], [-1.05, 1.05], color = 'maroon')
        ax0.plot([t2_end, t2_end], [-1.05, 1.05], color = 'maroon')
        ax0.set_xlabel("Time [s]", fontsize = 13)
        ax0.set_ylabel("Amplitude", fontsize = 13)
        ax0.set_xlim([0, T])
        ax0.set_ylim([-1.05, 1.05])
        
        # 窓処理してFFTスペクトルとスペクトル包絡を計算
        [freq, S1, yc1] = FFTspectrum_cepstrum(y1, fs, winSize)
        [freq, S2, yc2] = FFTspectrum_cepstrum(y2, fs, winSize)
        
        [fqc, envC1] = envelope_c(yc1, liftSize)
        [fqc, envC2] = envelope_c(yc2, liftSize)
        
        # （左） 分析区間1のスペクトル
        ax1.cla()
        ax1.plot(freq, S1)
        if chkEnv.current.value == True:
            ax1.plot(fqc, envC1, color = 'green')
        ax1.set_title("Time-1 spectrum")
        ax1.set_xlabel("Frequency [Hz]", fontsize = 13)
        ax1.set_ylabel("Relative level [dB]", fontsize = 13)
        ax1.set_xlim([0, 4000])
        ax1.set_ylim([-55, 5])
        # （右） 分析区間2のスペクトル
        ax2.cla()
        ax2.plot(freq, S2)
        if chkEnv.current.value == True:
            ax2.plot(fqc, envC2, color = 'green')
        ax2.set_title("Time-2 spectrum")
        ax2.set_xlabel("Frequency [Hz]", fontsize = 13)
        ax2.set_ylabel("Relative level [dB]", fontsize = 13)
        ax2.set_xlim([0, 4000])
        ax2.set_ylim([-55, 5])
        
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
        
        ft.Text("分析したい時間点1:　　　分析したい時間点2:"),  # 分析区間指定のガイドテキスト
        
        # 分析区間指定ボックス（開始用）
        ft.Row(
            controls=[
                ft.TextField(               # 時間位置1
                    ref  = setWin1Time,
                    label = "時刻1 [秒]",
                    width = 100,
                ),
                ft.Text("　　　"),
                ft.TextField(               # 時間位置2
                    ref = setWin2Time,
                    label = "時刻2 [秒]",
                    width = 100,
                ),
                ft.Text("　　　"),
                ft.Checkbox(                # スペクトル包絡チェック
                    ref = chkEnv,
                    label = "スペクトル包絡"
                )
            ]
        ),
        
        # 分析ボタン
        ft.ElevatedButton("スペクトル分析", on_click = FFTButton_clicked),
        
        # 振幅スペクトル
        MatplotlibChart(fig, expand=True),
 
    )

ft.app(target=main)
