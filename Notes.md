# TODO:
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
