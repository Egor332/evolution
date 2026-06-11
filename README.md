# Eksperymenty Ewolucyjne – Instrukcja Uruchomienia

Niniejszy tekst opisuje strukturę i sposób uruchamiania eksperymentów badających ewolucyjne metody uczenia maszynowego (CMA-ES, L-SHADE) w porównaniu do klasycznego podejścia gradientowego (AdamW). Do eksperymentów wykorzystano zbiór danych **Wine Quality Dataset** w zadaniach klasyfikacji binarnej oraz wieloklasowej.

---

## 1. Opis skryptu `run_experiments.py`

Głównym punktem wejścia do przeprowadzania badań jest skrypt [run_experiments.py]. Automatyzuje on wykonanie pięciu różnych scenariuszy badawczych:

### Eksperyment 1: Reality Check (Weryfikacja Podstawowa)
* **Cel:** Weryfikacja poprawności implementacji modeli (CMA-ES, L-SHADE, AdamW) na sieci o rozmiarze ok. 500 parametrów przy zadaniu klasyfikacji binarnej.
* **Ziarna losowości (seeds):** `42, 43, 44` (3 niezależne uruchomienia).
* **Wynik:** Wykres porównawczy średniej precyzji (Precision) i czułości (Recall) zapisywany w `experiments_result/images/Exp1_Summary_Metrics.png`.

### Eksperyment 2: Parameters Amount (Wpływ Rozmiaru Sieci)
* **Cel:** Ewaluacja wpływu liczby parametrów sieci (100, 300, 500, 1000, 2000) na skuteczność modeli oraz ich czas obliczeniowy.
* **Ziarna losowości (seeds):** `100, 101, 102, 103, 104` (5 uruchomień dla każdej konfiguracji).
* **Wynik:** Wykresy zależności metryk i czasu od rozmiaru sieci zapisywane jako `Exp2_Precision_vs_Size.png`, `Exp2_Recall_vs_Size.png` oraz `Exp2_Time_vs_Size.png`.

### Eksperyment 3: Multiclass Prediction (Klasyfikacja Wieloklasowa)
* **Cel:** Test modeli w trudniejszym zadaniu klasyfikacji wieloklasowej (przewidywanie konkretnych ocen wina) przy użyciu optymalnego rozmiaru sieci wyznaczonego w Eksperymencie 2 (domyślnie 1000 parametrów).
* **Ziarna losowości (seeds):** `200` (1 uruchomienie).
* **Wynik:** Macierze pomyłek (Confusion Matrix) oraz wykresy F1-score dla poszczególnych klas zapisywane osobno dla każdego z algorytmów (np. `Exp3_ConfMatrix_cmaes.png`, `Exp3_F1_Scores_cmaes.png`).

### Eksperyment 4: Statistical Stability (Stabilność Statystyczna)
* **Cel:** Zbadanie powtarzalności i stabilności osiąganych wyników na bazie 15 niezależnych uruchomień na klasyfikacji binarnej.
* **Ziarna losowości (seeds):** Zakres od `300` do `314` włącznie (`range(300, 315)`).
* **Wynik:** Wykresy pudełkowe (Boxplots) metryk Precision i Recall zapisywane w plikach `Exp4_Precision_Stability.png` oraz `Exp4_Recall_Stability.png`.

### Eksperyment 5: Number of Hidden Layers (Wpływ Głębokości Sieci)
* **Cel:** Porównanie wydajności modeli dla sieci posiadających 1, 2 lub 3 warstwy ukryte, przy stałej łącznej liczbie ok. 1000 parametrów.
* **Ziarna losowości (seeds):** Zakres od `500` do `504` włącznie (`range(500, 505)`).
* **Wynik:** Wykresy zależności Precision i Recall od liczby warstw ukrytych, zapisywane jako `Exp5_Precision_vs_Layers.png` oraz `Exp5_Recall_vs_Layers.png`.

> [!IMPORTANT]
> **Uwaga dotycząca ziaren losowości:**
> Wszystkie ziarna losowości (seeds) użyte do inicjalizacji generatorów pseudolosowych (Python `random`, `numpy`, `torch`) są **zdefiniowane bezpośrednio wewnątrz kodu skryptu** w celu zagwarantowania pełnej powtarzalności badań i eliminacji konieczności ich ręcznego przekazywania w argumentach.

---

## 2. Przygotowanie środowiska

Przed uruchomieniem eksperymentów należy utworzyć wirtualne środowisko Pythona i zainstalować wymagane zależności z pliku `requirements.txt`.

> [!NOTE]
> Wymagany jest Python 3.10 lub nowszy.

### System Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### System Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Systemy Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> [!TIP]
> Plik `requirements.txt` zawiera konfigurację instalacji PyTorch z obsługą CUDA (cu132). Jeśli nie posiadasz karty graficznej NVIDIA, PyTorch automatycznie skorzysta z CPU.

---

## 3. Uruchamianie eksperymentów

Po aktywacji środowiska wirtualnego (patrz punkt 2) skrypt uruchamia się poleceniem:

```bash
python run_experiments.py
```

### Opcje CLI

Skrypt akceptuje opcjonalny argument `--exp`, który pozwala na uruchomienie wyłącznie wybranego eksperymentu (1-5). W przypadku braku podania argumentu uruchomione zostaną **wszystkie eksperymenty po kolei**.

* **Uruchomienie wszystkich eksperymentów (domyślne):**
  ```bash
  python run_experiments.py
  ```

* **Uruchomienie tylko Eksperymentu 1 (Reality Check):**
  ```bash
  python run_experiments.py --exp 1
  ```

* **Uruchomienie tylko Eksperymentu 3 (Multiclass):**
  ```bash
  python run_experiments.py --exp 3
  ```

---
