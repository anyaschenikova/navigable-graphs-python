# Navigable Graphs Python
Python based research tool for studying navigable graphs for nearest neighbour search

Using the SIFT dataset:
```
python navigable-graphs.py --dataset sift
```

Using synthetic data with 3D vectors:
```
python navigable-graphs.py --dataset synthetic --K 20 --k 5 --dim 3 --n 500 --nq 100 --ef 20 --M 2
```

# New

## График 1: сравнение с M0

Для начала я провела исследование относительно того, насколько сильно влияет параметр M0 в значении от 50 до 150 и выяснила, что мой алгоритм дает такой же recall на m0 = 150, при этом имея меньшее количество вычисления. 

![image](/images/m0_changing.png)

## График 2-3: сравнение recall vs calsulations

Затем я провела исследования на зависимость между recall и количеством вызовов функций distance.

Вывод: Это замер при m0 = 50 для быстроты времени работы. Замер показывает, что мой алгоритм достигает сравнимого качества при ef = 50, однако даже на это требуется больше количества операций. Однако если мы будем смотреть время выполнения функции add и соответствующего построения качества, то мы увидим колоссальную разницу в необходимом времени. То есть можно использовать мой алгоритм и достичь результат гораздо быстрее, при чуть худшем количестве вызовов расстояния.

![image](/images/recall%20vs%20calculations.png)

Я поставила M0 = 150.
Вывод: Мы можем достичь 1.0 при гораздо меньшем количестве операций, по сравнению с оригинальным алгоритмом.

![image](/images/recall_vs_calc_m0_150.png)