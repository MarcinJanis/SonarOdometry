# TODO:

o przeanalizowaniu pliku utils.py w połączeniu z poprzednią logiką z graph_train.py i dpso_train.py, mamy już pełny obraz. 
Dobra wiadomość jest taka, że matematyka w utils.py jest poprawna dla układu prawoskrętnego (jakim jest NED). 

Zła wiadomość jest taka, że potwierdza to całkowitą niespójność znaków między modułami. 
Oto szczegółowe zestawienie miejsc, które musisz poprawić, aby system poprawnie odzwierciedlał konwencję NED (gdzie dodatnie $Z$ to dół/głębokość, a dodatni pitch to pochylenie w dół).


# 1. Znak kąta elewacji ($\phi$) a transformacje kartezjańskie
Diagnoza:
W utils.py w funkcji transform_polar2cart masz równanie: z = r * torch.sin(phi).
Funkcja depth_to_elev_angle liczy kąt jako: gt_elevation = torch.asin(depth/r).
Skoro depth (głębokość na dno) w NED jest dodatnie, to gt_elevation wyjdzie dodatnie. 
Skoro $\phi$ jest dodatnie, to z równania na z współrzędna $Z$ również wyjdzie dodatnia (czyli zgodnie z założeniami wyląduje na dnie).
Problem: W pliku graph_train.py wymuszałeś ujemne $\phi$ (odrzucając minus przed parametrem pitch, np. - sonar_cfg.position.pitch). 
Powoduje to, że sieć i graf w graph_train.py szukają punktów nad sonarem (ujemne $Z$), podczas gdy dpso_train.py za pomocą utils.py każe im optymalizować się do punktów na dnie (dodatnie $Z$).

Jak to naprawić (w graph_train.py):
Musisz usunąć minusy i upewnić się, że środek wiązki celuje w dół na plusie.Python# Jeśli pozycja pitch określa środek wiązki w dół (np. 0.2 rad):
self.phi_max = sonar_cfg.position.pitch + (self.fov_vertical / 2.0)
self.phi_min = sonar_cfg.position.pitch - (self.fov_vertical / 2.0)

(Uwaga: założyłem tu podział FOV na pół wokół kąta pitch, bo fov to zazwyczaj całkowity kąt widzenia)

2. Globalny vs Lokalny kąt elewacji przy walidacji FOV

Diagnoza:W utils.py funkcja project_points działa w 100% poprawnie: 
bierze punkt w układzie sferycznym ramki źródłowej -> przerzuca do kartezjańskiego -> transformuje do przestrzeni globalnej -> transformuje do ramki docelowej -> przerzuca z powrotem na układ sferyczny.

Oznacza to, że punkt tgt_cooords zwracany przez tę funkcję w graph_train.py (w metodzie corr) jest wyrażony w lokalnym układzie współrzędnych sonaru docelowego.

Problem: Weryfikujesz ten lokalny kąt w graph_train.py za pomocą globalnych limitów z uwzględnieniem pitcha: Pythonout_of_range = out_of_range | (tgt_cooords[:,2] > self.phi_max + coords_eps) # ŹLE
Lokalny układ współrzędnych sonaru "nie wie", że jest pochylony. Dla niego środek wiązki to $\phi = 0$.

Jak to naprawić (w graph_train.py):
Walidacja odrzucająca punkty poza zasięgiem pola widzenia w metodzie corr powinna wyglądać symetrycznie wokół zera:Python# FOV definiuje lokalny widok głowicy
phi_max_local = self.fov_vertical / 2.0
phi_min_local = -self.fov_vertical / 2.0

out_of_range = out_of_range | (tgt_cooords[:,2] > phi_max_local + coords_eps)
out_of_range = out_of_range | (tgt_cooords[:,2] < phi_min_local - coords_eps)


3. Założenie płaskiego dna w depth_to_elev_angle
Diagnoza:W pliku utils.py dla funkcji depth_to_elev_angle napisałeś wprost komentarz: # Note: Assumption, that there is flat surrounding!.

Podejście torch.asin(depth_r_ratio) jest w 100% zgodne matematycznie (przeciwprostokątna to $r$, przyprostokątna to $depth$ czyli współrzędna $Z$).

Miejsca do wyjaśnienia/pilnowania:
Czy podawane do sieci depth_gt to faktycznie "współrzędna Z od sonaru do danego piksla", czy "wysokość pojazdu nad płaskim dnem (odczytana z DVL/echosondy)"?
Jeśli to to drugie, a dno bywa strome, model ground truth będzie nie do końca poprawny. 

Do algorytmu reprojekcji i Bundle Adjustment dla konkretnych patchów potrzebujesz dokładnej współrzędnej lokalnego $Z$ dla tego piksla. 

Zostaw to tak jak jest, ale miej świadomość, że jeśli nakarmisz to danymi z bardzo nierównego dna morskiego, loss ground-truth (w przypadku treningu nadzorowanego) będzie uczył sieć fałszywej geometrii.

4. Matematyka Kwaternionów a układ prawoskrętny (RHS)

Diagnoza:NED jest w pełni prawoskrętnym układem kartezjańskim (Right-Handed System). Zaimplementowany przez Ciebie produkt Hamiltona (hamilton_product) i koniugacja (q_conjugate) w utils.py operują jako standardowe obroty $q \otimes v \otimes q^{-1}$, co idealnie odpowiada prawoskrętnemu układowi, przy założeniu konwencji [x, y, z, w].

Zgodność: Tu jest całkowita spójność, pod warunkiem, że biblioteka pypose, której używasz w bundle_adjustment.py (obiekt pp.SE3), korzysta dokładnie z tej samej konwencji ułożenia komponentów kwaternionu (czyli w na końcu - skalarny).

Zwróć uwagę: Niektóre biblioteki graficzne i robotyczne przyjmują kwaternion w formacie [w, x, y, z]. 
Sprawdź w dokumentacji PyPose, czy przyjmuje [x, y, z, w]. 

Jeśli nie, transformacje wewnątrz optymalizatora BA się posypią. 

PyPose standardowo używa konwencji [x, y, z, w], więc powinno być okej, ale warto mieć to z tyłu głowy, jeśli błędy nadal by występowały.

Podsumowanie
Dzięki wglądowi w utils.py widać wyraźnie, że matematyka jest solidna i radzi sobie z NED bardzo dobrze. Cały problem leżał w dwóch linijkach pliku graph_train.py, gdzie niepotrzebnie odwróciłeś znak globalnego kąta, wprowadzając schizofrenię w domenie elewacji. 

Wprowadź korekty w punktach 1 i 2, a geometria całego układu zamknie się w spójną, poprawną całość.

<!-- 

!! Problem: 
- valid mask zmeinic na float, 
- nie wyrzucac niby krawedzi, przetwarzac wszystkie
- korelacje albo wagi, idk jeszcze przemnozyc przez valid mask 



- Dodać F.normalize zaraz przed zwróceniem predykcji póz
  
- Dodać F.normalize za każdym razem, gdy modyfikuje pozę, czyli w zasadzie w BA
  
- zrobić nowe dataset visualisation, używając modułu do ładowania danych w trybue inferencji, z opcją transformacji do układu kartezjańskiego. Do tego, dodać opcję wizualizwoania key points, zeby zobaczyc, czy jest keypoints śledzą te same miejsca.

- zaimplementować alternatywny sposób wybierania keypoitns. Jest to alternatywna forma "augumentacji", ponieważ zrobi ekstrakcje innych punktów. -->




Inference mode DPSO:


- zwracać ma pozycję estymowaną, pierwszy raz
- opcjonalnie zwraca

- niech jako opcjnalne wyjście wyliczy i zwraca:
słownik: numer patcha globalny, listę klatek na których się patch pojawia (id globalne), współrzędne na każdejz klatek po estymacji/poprawkach.
czyliL i, j oraz patch_coords tylko w posegregowanej ładniejszej formie. Można dorzucić też pewność/wagę.

- opcjonalne zapisywanie do pliku: pozycji estymowanej pierwzy raz, pozycji usuwnaej z bufora. 
- opcjonalne zapisywanie do pliku punktów 3d wyestymowanych wtórnie.











## Ideas:


wykres execution time (śrendi) od ilości punktów. 

1) naprawić zapiswanie trajektor estymowanej
2) niech zapisuje do dwóch różnych plików, pierwotna i wtóna estymacja 

3) Ustawianei początkowej pozycji w inferencji
