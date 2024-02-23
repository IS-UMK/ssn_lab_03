# SSN. Lab. 3. Sieć MLP

Zapoznaj się z zawartością notatnika Jupyter umieszczonego w repozytorium  i wykonaj zawarte w nim ćwiczenia.

Notatnik: [03_mlp_softmax.ipynb](https://github.com/IS-UMK/ssn_lab_03/blob/master/03_mlp_softmax.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IS-UMK/ssn_lab_03/blob/master/03_mlp_softmax.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IS-UMK/ssn_lab_03/master?filepath=03_mlp_softmax.ipynb)

---

## Zad. 3. MLP

Zaimplementuj sieć MLP z jedną warstwą ukrytą przeznaczoną do klasyfikacji wieloklasowej ($c$ klas) uczoną za pomocą wstecznej propagacji.  
Sieć działa dla danych o specyfikacji:
* $X$ zbiór treningowy (macierz $n \times d$), $n$ przypadków opisanych $d$ zmiennymi
* $\mathbf{y} \in [0, 1, 2, \ldots, c-1 ]$ wektor etykiet klas

Sieć uczona jest zgodnie z poniższym algorytmem realizowanym przez metodę ``fit(X, y)``, której implementację znajdziesz w pliku [MLPClassifier.py](https://github.com/IS-UMK/ssn_lab_03/blob/master/MLPClassifier.py). Zaimplementuj brakujące metody ``init()``, ``feedforward()``, ``backprop()``, ``update()``, ``predict()`` tak aby zrealizować uczenie oraz predykcję modelu. 

**Algorytm uczenia**  
Parametry początkowe (ustawiane w konstruktorze): 
* $N$ lość epok
* $k$ ilość neuronów w warstwie ukrytej
* $\eta$ krok uczenia

1. Zamień etykiety $y$ do postaci wektora _one-hot_ $\mathbf{y}_{onehot}$
2. Zainicjuj macierze wag i wektory wyrazów wolnych (metoda ``init(X, y)``)<br>
   warstwa ukryta: $\mathbf{W}_1$ (wymiary  $k \times d$), \, $\mathbf{b}_1$ (wymiary $k \times 1$) <br>
   warstwa wyjściowa: $\mathbf{W}_2$ (wymiary $c \times k$),\, $\mathbf{b}_2$ (wymiary $c \times 1$)
3. Powtarzaj $n$ razy:
4. <ul>Dla każdego $\mathbf{x}$ ze zbioru treningowego wykonaj</ul>
5. <ul><ul>propagacja sygnału (metoda forward(x))<br>  
   $\mathbf{h}_1 = \sigma \left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right)$  <br>
   $\mathbf{h}_2 = \operatorname{softmax} \left(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2\right)$  <br>
   gdzie $\sigma$ funkcja aktywacji warstwy ukrytej $\sigma(x)=\frac{1}{1+e^{-x}}$ </ul></ul>
6. <ul><ul>oblicz sygnał błędu (metoda backward(y)) <br> 
   $\boldsymbol{\delta}_2 = \mathbf{y}_{onehot} - \mathbf{h}_2$<br>
   $\boldsymbol{\delta}_1 = \sigma' \circ \boldsymbol{\delta}_2 \, \mathbf{W}_2$<br>
   gdzie $\sigma'$ to pochodna funkcji aktywacji $\sigma^{\prime}(x)=\sigma(x)(1-\sigma(x))$, iloczyn $\mathbf{a}\circ \mathbf{b}$ jest wykonywany po współżędnych </ul></ul>
7. <ul><ul>aktualizacja wag  (metoda update(x))<br>
   $\mathbf{W}_1 \leftarrow \mathbf{W}_1 + \eta \, \boldsymbol{\delta}_1 \cdot \mathbf{x}^T$  <br>
   $\mathbf{b}_1 \leftarrow \mathbf{b}_1 + \eta \, \boldsymbol{\delta}_1$  <br>
   $\mathbf{W}_2 \leftarrow \mathbf{W}_2 + \eta \, \boldsymbol{\delta}_2 \cdot \mathbf{h}_1^T$   <br>
   $\mathbf{b}_2 \leftarrow \mathbf{b}_2 + \eta \, \boldsymbol{\delta}_2$  <br> iloczyn $\mathbf{a}\cdot \mathbf{b}^T$ oznacza iloczyn zewnętrzny </ul></ul>
8. <ul>na koniec epoki oblicz i zachowaj wartość funkcji kosztu entropii krzyżowej $J$ oraz poprawność klasyfikacji  </ul>
   $$J = -\sum_{i=1}^{n}\mathbf{y}_{i} \log{f(\mathbf{x}_i})$$ 

Podziel zbiór danych ``digits`` na część treningową i testową w proporcji 50%/50%. Wykonaj trening na zbiorze treningowym. Wyrysuj wykres funkcji kosztu oraz poprawności klasyfikacji w kolejnych epokach. Wyznacz poprawność klasyfikacji na zbiorze testowym. 
Spróbuj dobrać parametry (ilość epok, ilość neuronów, stała uczenia) tak aby uzyskać jak najlepszy wynik.  

Rozwiązanie w postaci notatnika Jupyter (``.ipynb``) lub skrypt w języku Python (``.py``) umieść w Moodle lub prześlij do repozytorium GitHub.

---
## Materiały:

* S. Raschka, "Python machine learning"  
  Rozdział 12: raining Artificial Neural Networks for Image Classification"  
  https://github.com/rasbt/python-machine-learning-book/tree/master/code/ch12
* J. Vitay, "Neurocomputing"
  Rozdział 8: "Multi-layer perceptron"  
  https://julien-vitay.net/lecturenotes-neurocomputing/5-exercises/ex8-MLP.html
* Jay Mody, "Numerically Stable Softmax and Cross Entropy,  
  "https://jaykmody.com/blog/stable-softmax/




