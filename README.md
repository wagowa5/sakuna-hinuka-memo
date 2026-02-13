# sakuna-hinuka-memo
サクナヒヌカめも

# 敵防御力推定

通常攻撃（係数0.25）、7個揃っている場合

```
python estimate_def.py --atk 2219 --mult 0.25 --damages 521,527,532,537,543,548,554
```

スキル（係数1.6）、観測が一部しかない場合

```
python estimate_def.py --atk 1522 --mult 1.6 --damages 1520,1535,1567,1582,1614
```

防御Down 16.5%のログから「元の防御力」も出したい場合

```
python estimate_def.py --atk 2002 --mult 0.25 --damages 413,417,421,426,430,434,438 --defdown 0.165
```

# 推測ダメージ出力

```
python dmg_calc.py --atk 2219 --mult 0.25 --def 780
```
