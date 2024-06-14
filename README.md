# py_acoustProcesssingTools
小澤・鳥谷研　音響基礎の基礎

【はじめる前に】
numpy, scipy, matplotlib, soundfile, sounddevice, flet ライブラリが必要．
なければ > pip install ○○○○ で追加する．

【使用方法】
manual_distr.pdf の p.23~25, 34-35 を参照のこと．

【音の出し入れがうまくいかなかったら…】
最初に sd_chkDefaultDevice.py を実行し，入出力デバイス番号を確認の上，
各プログラムの sd.default.device = [ ] の中に適切な入出力デバイス番号を設定する．
