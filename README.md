# SSN. Lab. 3. Sieć MLP

Zapoznaj się z zawartością notatnika Jupyter umieszczonego w repozytorium  i wykonaj zawarte w nim ćwiczenia.

Notatnik: [03_mlp_softmax.ipynb](https://github.com/IS-UMK/ssn_23_lab_01/blob/master/03_mlp_softmax.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IS-UMK/ssn_23_lab_01/blob/master/03_mlp_softmax.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IS-UMK/ssn_23_lab_01/master?filepath=03_mlp_softmax.ipynb)

---

## Zad. 2. MLP


Zaimplementuj sieć MLP z jedną warstwą ukrytą przeznaczoną do klasyfikacji wieloklasowej ($c$ klas) uczoną za pomocą wstecznej propagacji wg. poniższego schematu.

Dane: $X$ zbiór treningowy opisany $d$ zmiennymi, $y = \{0, 1, 2, \ldots, c-1\}$ wektor etykiet klas

Parametry (ustawiane w konstruktorze): ilość epok $n$, ilość neuronów w warstwie ukrytej $k$, krok uczenia $\eta$

Zaimplementuj metodę ``fit(X, y)`` realizującą uczenie sieci algorytmem wstecznej propagacji zgodnie z poniższym algorytmem:

1. Zamień etykiety $y$ do postaci wektora _one-hot_ $\mathbf{y}_{onehot}$
2. Zainicjuj macierze wag i wektory wyrazów wolnych <br>
   $\mathbf{W}_1$ wymiar  $k \times d$<br>
   $\mathbf{b}_1$ wymiary $k \times 1$ <br>
   $\mathbf{W}_2$ wymiar  $c \times k$<br>
   $\mathbf{b}_2$ wymiary $c \times 1$
3. Powtarzaj $n$ razy:
4. Dla każdego $\mathbf{x}$ ze zbioru treningowego $\mathbf{X}$ wykonaj
5. propagacja sygnału (zaimplementuj metodę ``forward(x)``)  
   $\mathbf{h}_1 = \sigma \left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right)$  <br>
   $\mathbf{h}_2 = \sigma \left(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2\right)$  
   gdzie funkcja aktywacji warstwy ukrytej<br>
   $\sigma(x)=\frac{1}{1+e^{-x}}, \qquad \sigma^{\prime}(x)=\sigma(x)(1-\sigma(x))$<br>
   funkcja wyjściowa softmax <br>
   $f(\mathbf{x})_i=\frac{e^{-x_i}}{\sum e^{-x_i}}$
6. oblicz sygnał błędu (metoda ``backward(y)``)  
   $\boldsymbol{\delta}_2 = \mathbf{y}_{onehot} - \mathbf{h}_1$<br>
   $\boldsymbol{\delta}_1 = \sigma' \, \boldsymbol{\delta}_2 \, \mathbf{W}_2$
7. aktualizacja wag  
   $\mathbf{W}_1 \leftarrow \mathbf{W}_1 + \eta \, \boldsymbol{\delta}_1 \cdot \mathbf{x}^T$  
   $\mathbf{b}_1 \leftarrow \mathbf{b}_1 + \eta \, \boldsymbol{\delta}_1$  
   $\mathbf{W}_2 \leftarrow \mathbf{W}_2 + \eta \, \boldsymbol{\delta}_2 \cdot \mathbf{h}_1^T$  
   $\mathbf{b}_2 \leftarrow \mathbf{b}_2 + \eta \, \boldsymbol{\delta}_2$  
8. na koniec epoki oblicz i zachowaj wartość funkcji kosztu $L$ oraz poprawność klasyfikacji  
   $L = - \sum_{i=1}^{n}\mathbf{y}_{i} \log{f(\mathbf{x}_i})$  


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




