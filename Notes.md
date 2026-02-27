# TODO:

1) Add to Graph following funciton
   
 - PatchGraph.get_position()
In inference mode it shound return newest position (for latest added frame)
In trening mode: idk yet, it should return everything to loss fcn. Maybe first estimation and last estimation 

 - PatchGrapg.update_pose(optimized_pose, optimized_phi)
Get optimized poses and phi, assign to graph.

- PatchGrapg.get_hidden_state(patch_idx)
Only in inference mode, get hidden state for this patches that are included in current edges 

- PatchGrapg.save_hidden_state(h, patch_idx)
When new hidden state returned save it for propper patches 



Elementy do zaadoptowania z DPVO `https://github.com/princeton-vl/DPVO`.


Do poprawy: 
_____________________

# 1.  Zabicie przepływu gradientów w Bundle Adjustment (Trening)

W architekturach RAFT/DPVO siła tkwi w możliwości różniczkowania przez pętlę optymalizacji, 
dzięki czemu błąd na końcu propaguje się wstecz do sieci wyliczającej delta i weights. 

W pliku bundle_adjustment.py na samym końcu metody run zwracasz wyciągnięte parametry używając .detach():

Python:

optimized_poses = self.poses.detach()
optimized_elevation = self.elevation_angle.detach()
return optimized_poses.tensor().view(self.b, self.n, 7), ...

To całkowicie odcina graf obliczeniowy. Model nie będzie się uczył. 

Zamiast tego powinieneś użyć wbudowanych mechanizmów optymalizatorów z biblioteki pypose, 
które wspierają różniczkowanie analityczne lub wyliczają odpowiedź różniczkzkowalną wewnątrz frameworka, 
i zwracać tensory bez odpinania ich od grafu obliczeniowego podczas treningu.

# 2. Brak sprzężenia zwrotnego korelacji (Static Correlation)

DPVO iteracyjnie poprawia estymaty. W każdej iteracji wektor ukryty ($h$) i estymata pozy aktualizują się, 
po czym sieć powinna ponownie wyciągnąć cechy z mapy korelacji (tzw. lookup) bazując na nowej rzutowanej pozie.

W Twoim pliku dpso.py funkcja update_step() wyciąga corr tylko raz, przed pętlą for _ in range(self.update_iter):
:Python

corr, ctx, ... = self.PatchGraph.update_step(...) # Wywoływane RAZ
for _ in range(self.update_iter):
    self.h, correction = self.UpdateOperator(h, None, corr, ctx, ...) # Corr jest STATYCZNE
    ...


Oznacza to, że UpdateOperator dostaje dokładnie tę samą mapę korelacji w każdej iteracji, 
mimo że pozy i przewidywane poprawki się zmieniają. Sieć działa na "ślepo" po pierwszej iteracji, 
co całkowicie psuje ideę podejścia opartego na RAFT. Musisz aktualizować pozycję wewnątrz pętli 
i przepróbkowywać (re-sample) corr w każdym kroku.


# 3. Niespójności między train a inference (Domain Gap)
Wykryłem różnice w zachowaniu systemu, które sprawią, że sieć nauczona w trybie treningowym będzie działać źle w środowisku produkcyjnym (inference).

## Model ruchu i inicjalizacja (Priors)

Inference (graph_inference.py): 
Funkcja approx_movement przewiduje nową pozę na podstawie poprzedniej prędkości z użyciem fizycznego modelu (np. ruchu liniowego dla $SE(3)$). 
Sieć ma za zadanie jedynie skorygować ten domyślny ruch (delta).

Trening (graph_training.py): 
Funkcja approx_movement inicjalizuje całą serię póz jako zera (lub macierze identyczności).
Skutek: W treningu sieć uczy się przewidywać absolutny ruch (od 0 do wartości rzeczywistej), 
podczas gdy w inferencji wymaga się od niej korekty małego błędu między przewidywaniem liniowym a rzeczywistością. 
Inicjalizacja w obu trybach musi być tożsama. W trybie treningowym musisz zasymulować ten sam mechanizm przewidywania linear, 
korzystając z danych wejściowych Ground Truth w odpowiedni sposób.


## Maskowanie krawędzi (Dynamic vs Static Batching)

W funkcji corr pliku graph_training.py używasz warunku eps=0.1 do weryfikacji pola widzenia (FOV) dla valid_mask, po czym filtrujesz tensory np. valid_j = self.j[valid_mask].

W graph_inference.py nie używasz eps (co sprawia, że krawędzie zachowują się ostrzej).
Ponadto, ucinanie maską rozmiarów tensorów w trakcie działania batcha treningowego potrafi zniszczyć równe wymiary batcha, 
jeśli dla różnych serii odpadnie inna liczba krawędzi. 
Rekomenduję zachowanie pełnych tensorów i nakładanie maski mnożąc wagi zaufania (weights) dla nieprawidłowych krawędzi przez 0.


## Mniejsze błędy i luki logiczne
Zarządzanie stanem pozy w dpso.py: 
W trybie treningowym tworzysz nową instancję BundleAdjustment w każdej iteracji (BA = BundleAdjustment(poses, ...)), przesyłając zmienną poses. 
Jednak poses nie jest nigdzie nadpisywana wynikiem z poprzedniej iteracji BA (opt_poses). 

Twój optymalizator za każdym razem startuje od zera.

Literówka w nazewnictwie: Importujesz klasę jako Graph_interference zamiast Graph_inference (from .graph_inference import Graph as Graph_interference). 
Zwyczajowa nazwa to "inference" (wnioskowanie), a nie "interference" (zakłócenie/interferencja).


Proponowane rozwiązanie

Twoim priorytetem powinno być naprawienie głównej pętli sprzężenia zwrotnego. 

Pętla w DPSO.forward (plik dpso.py) powinna wyglądać koncepcyjnie tak:

Pobierz ramkę i wyznacz początkowe pozycje/położenia łat.

Inicjalizuj wektor ukryty $h$.

Pętla dla update_iter:

Wylicz rzutowania i wyciągnij corr na podstawie aktualnie przechowywanych pozy/kąta.

Wykonaj UpdateOperator, który zwraca optyczne delty i wagi ufności.

Zainicjuj BA z nowymi deltami.Uruchom optymalizator BA, który zwraca nowe poses.

Zastąp stare poses nowymi poses na następną iterację.

Czy chciałbyś, abym napisał dla Ciebie zrefaktoryzowaną strukturę głównej pętli forward w dpso.py, 
upewniając się, że mapa korelacji jest poprawnie przepróbkowywana, 
a wagi nie tracą gradientów?


