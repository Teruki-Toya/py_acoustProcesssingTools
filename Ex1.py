# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:27:03 2023

@author: Teruki Toya
"""

# %%
import flet as ft
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart
import scipy.fftpack as spfft
from scipy import signal

# 固定パラメータの定義 ---------------
sd.default.device = [1, 3] # Input, Outputデバイスを指定

fs = 44100      # サンプリングレート [samples/sec]
T = 1.0         # 信号長 [sec]
N = round(fs*T) # 信号のサンプル数 [samples]

########## ここから関数群 ##########

## FFTで振幅スぺクトルを計算する関数
def FFTspectrum(x, fs, fftSize):
  
  # 高速フーリエ変換（FFT）
  X = spfft.fft(x, fftSize) / fftSize
  
  # 分析対象の周波数列
  idx = np.linspace(0, int(fftSize/2), int(fftSize/2+1)) # 周波数ビン番号
  freq = idx * fs / fftSize  # 各周波数ビンの対応する周波数
  
  # 振幅スペクトル
  S_amp = 20 * np.log10(abs(X))  # 各周波数ビンにおける相対振幅スペクトル
  S_amp = S_amp[0 : len(freq)] + 6  # 図の辻褄合わせで補正（+12 dB）
  
  return freq, S_amp

########## ここまで関数群 ##########

# Flet の処理 ---------------------
def main(page):
    
    page.title = "音信号の生成と観察"  # タイトル
    # page.vertical_alignment = ft.MainAxisAlignment.CENTER

    page.window_width = 750  # 幅
    page.window_height = 750  # 高さ
    page.window_top = 100  # 位置(TOP)
    page.window_left = 100  # 位置(LEFT)
    page.window_always_on_top = True  # ウィンドウを最前面に固定
    page.window_center()  # ウィンドウをデスクトップの中心に移動
    
    setSnd_rg = ft.Ref[ft.RadioGroup]()   # 音種ラジオボタングループを定義
    
    setA_slider = ft.Ref[ft.Slider]()   # 振幅設定スライドバーを定義
    
    txtFrq = ft.Ref[ft.Text]()      # 周波数設定部テキスト表示を定義
    
    setF0 = ft.Ref[ft.TextField]()  # 基本周波数（f0）入力ボックスを定義
    setF1 = ft.Ref[ft.TextField]()  # 高調波周波数1（f1）入力ボックスを定義
    setF2 = ft.Ref[ft.TextField]()  # 高調波周波数2（f2）入力ボックスを定義
    
    fig = plt.figure(figsize=(10, 4))  # 図示
    # （左） 時間波形
    ax1 = fig.add_subplot(121)
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time [s]", fontsize = 13)
    ax1.set_ylabel("Amplitude", fontsize = 13)
    ax1.set_xlim([0.485, 0.515])
    ax1.set_ylim([-1.05, 1.05])
    #（右） FFT振幅スペクトル
    ax2 = fig.add_subplot(122)
    ax2.set_title("Amplitude spectrum")
    ax2.set_xlabel("Frequency [Hz]", fontsize = 13)
    ax2.set_ylabel("Relative level [dB]", fontsize = 13)
    ax2.set_xlim([0, 10000])
    ax2.set_ylim([-35, 5])
    
    # 音種ラジオボタン変更時の動作を記述 -------------------------------
    def rg_changed(e):
        
        # 正弦波モードが選択されたとき:
        if setSnd_rg.current.value == "sin":
            txtFrq.current.value = "構成する周波数 f0, f1, f2 [Hz]:"
            setF1.current.visible = True  # f1・f2のボックスを見えるようにする
            setF2.current.visible = True
        
        # 正弦波モード以外が選択されたとき:
        else:
            txtFrq.current.value = "基本周波数 f0 [Hz]:"
            setF1.current.value = "" # f1・f2値をリセットする
            setF2.current.value = ""
            setF1.current.visible = False # f1・f2のボックスを見えなくする
            setF2.current.visible = False
        
        # ページを更新
        page.update()

    # 信号生成ボタン押下時の動作を記述 -----------------------------------
    def button_clicked(e):
        # パラメータ設定
        t = np.arange(N) / fs       # 信号の時間サンプル配列

        Ar = setA_slider.current.value   # 相対振幅レベル（振幅'1'を基準）
            
        if not setF0.current.value: # 純音周波数（ボックスから取得）
            f0 = 0
        else:
            f0 = int(setF0.current.value)
            if f0 >= fs/2:
                f0 = 0
                
        if not setF1.current.value:
            f1 = 0
        else:
            f1 = int(setF1.current.value)
            if f1 >= fs/2:
                f1 = 0
            
        if not setF2.current.value:
            f2 = 0
        else:
            f2 = int(setF2.current.value)
            if f2 >= fs/2:
                f2 = 0
            
        f = [f0, f1, f2]

        phi = 2 * np.pi * np.random.rand(3) - np.pi      # ランダム位相
        
        # 正弦波の生成
        if setSnd_rg.current.value == "sin":
            y_all = np.zeros((3, N))    # 3つの正弦波を格納する配列（初期値0埋め）
            for k in range(3):
                if f[k] == 0:   # 信号生成時の初期振幅 A_ini
                    A_ini = 0       # 生成しないとき -> 無音（'0'埋めの信号）
                else:
                    A_ini = 0.97    # 生成するとき -> 初期振幅1（念のため3%ダウン）
                # 正弦波を生成する：x = A sin(2πft + φ)
                x = A_ini * np.sin((2*np.pi*f[k]) * t + phi[k])
                y_all[k, :] = x;
                
            y_row = np.sum(y_all, axis=0)   # 行方向に信号を加算
        
        # 鋸波の生成
        elif setSnd_rg.current.value == "saw":
            if f[0] == 0:       # 信号生成時の初期振幅 A_ini
                A_ini = 0           # 生成しないとき -> 無音（'0'埋めの信号）
            else:
                A_ini = 0.97        # 生成するとき -> 初期振幅1（念のため3%ダウン）
            # 鋸波を生成する：x = A * sautooth(2πft)
            y_row = A_ini * signal.sawtooth((2*np.pi*f[0]) * t)
        
        # 矩形波の生成
        else:
            if f[0] == 0:
                A_ini = 0           # 生成しないとき -> 無音（'0'埋めの信号）
            else:
                A_ini = 0.97        # 生成するとき -> 初期振幅1（念のため3%ダウン）
            # 矩形波を生成する：x = A * square(2πft)
            y_row = A_ini * signal.square((2*np.pi*f[0]) * t)
        
        N_taper = round(0.1 * fs)   # テーパー部のサンプル数
        taper = np.sin(np.linspace(0, np.pi/2, N_taper)) # 上昇部
        taper = np.concatenate([taper, np.ones(N - 2*N_taper), np.flipud(taper)])
                                    # 下降部は上昇部の時間反転数列
        y = y_row * taper           # 生成波形にテーパー処理を施す
        if np.amax(np.abs(y)) != 0:
            y = (0.97 * (10**(Ar/20))) * y / np.amax(np.abs(y)) # 振幅調整
            
        # FFTしてスペクトル分析
        [freq, S_amp] = FFTspectrum(y, fs, len(y))
        
        # 描画 － （左）時間波形
        ax1.cla()
        ax1.plot(t, y)
        ax1.set_title("Waveform")
        ax1.set_xlabel("Time [s]", fontsize = 13)
        ax1.set_ylabel("Amplitude", fontsize = 13)
        ax1.set_xlim([0.485, 0.515])
        ax1.set_ylim([-1.05, 1.05])
        
        # 描画 － （右）FFT振幅スペクトル
        ax2.cla()
        ax2.plot(freq, S_amp)
        ax2.set_title("Amplitude spectrum")
        ax2.set_xlabel("Frequency [Hz]", fontsize = 13)
        ax2.set_ylabel("Relative level [dB]", fontsize = 13)
        ax2.set_xlim([0, 10000])
        ax2.set_ylim([-35, 5])
        
        # 音を出力する
        if setSnd_rg.current.value == "sin":
            A_play = 10**(-30/20)
        else:       # 鋸波と矩形波は「聴感上」うるさいので、わざと 10 dB落として鳴らす
            A_play = 10**(-40/20)
        
        sd.play((A_play *y), fs)
        
        #setF0.current.value = ""    # f0 ~ f2 の値を一旦空白にする
        #setF1.current.value = ""
        #setF2.current.value = ""
        page.update()               # ページを更新

    # Flet コントロールの追加とページへの反映 -------------------------------
    page.add(
        ft.Text("信号の種類:"),              # 振幅設定部テキスト
        ft.RadioGroup(                      # 音種ラジオボタン
            ref = setSnd_rg,
            content=ft.Row([
                ft.Radio(                       # 正弦波用
                    value = "sin",
                    label = "正弦波",
                    autofocus = True
                ),
                ft.Radio(                       # 鋸波用
                    value = 'saw',
                    label = "鋸波",
                ),
                ft.Radio(                       # 矩形波用
                    value = "sq",
                    label = "矩形波"
                )
            ]),
            on_change = rg_changed
        ),
        ft.Text(""),                        # 空行
        
        ft.Text("振幅 A（1を基準とした dB）:"),              # 振幅設定部テキスト
        ft.Slider(                          # 振幅設定スライドバー
            ref = setA_slider,
            min = -24,
            max = 0,
            divisions = 4,
            label = "{value} dB",
            width = 300
        ),
        ft.Text(""),                    # 空行
        
        ft.Text(                        # 周波数設定部テキスト
            ref=txtFrq,
            value = "構成する周波数 f0, f1, f2 [Hz]:"
        ),
        
        ft.Row(                         # 周波数入力ボックス
            controls=[
                ft.TextField(               # f0 用
                    ref  = setF0,
                    label = "周波数 f0",
                    width = 100,
                ),
                ft.TextField(               # f1 用
                    ref = setF1,
                    label = "周波数 f1",
                    width = 100,
                ),
                ft.TextField(               # f2 用
                    ref = setF2,
                    label = "周波数 f2",
                    width = 100,
                ),
            ]
        ),
        
        # 信号生成ボタン
        ft.ElevatedButton("信号を生成", on_click = button_clicked),
        
        # 時間波形
        MatplotlibChart(fig, expand=True),
    )

ft.app(target=main)

# %%
