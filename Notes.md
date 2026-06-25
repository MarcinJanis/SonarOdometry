Dataset jest do pobrania na dysku: https://drive.google.com/drive/u/2/folders/1BgosYlaRkQkSa43Jpgb6hoGK88n3bXLL.

Są tam dwa pliki: pełen dataset (SonarOdometryDataset.tar.gz) oraz mniejsza próbka do testowania (SonarOdometryDataset_sample.zip)

Wizualizacja reprojekcji 

**/notebooks/test/graph_training_test.ipynb**

Najpierw pod nagłówkiem: Reprojection for ground truth data jest reprojekcja punktów na bazie pozycji z ground truth, 
więc punkty powinny być dopasowane do siebie idealnie, jednak w większości przypadków reprojekcja jest zaburzona przez to że trzecia współrzędna punktów czyli kąt nachylenia wiązki sonaru jest jedynie szacowana na podstawie wysokości robota nad dnem. 

Później pod nagłówkiem: Reprojection for model predictions, jest reprojekcja na podstawie wstępnie zainicjalizowanych póz i do niej dodana jest poprawka do przepływu optycznego zwrócona przez sieć. Przez to że te poprawki to jest kilka pikseli, ciężko jest ocenić na oko, dlatego komórkę wyżej niż wizualizacja wypisywane są statystki dla wybranych patchy, m. in. ile wynosi poprawka zwrócona przez sieć (delta) oraz ile wynosi błąd reprojekcji po uwzględnieniu poprawki z sieci. 

Na koniec zwizualizowana jest trajektoria poprawiona przez bundle adjustment na podstawie poprawek z sieci. Użyte jest jednak bundle adjustment które nie radzi sobie za dobrze nawet na dokładnych danych wejściowych. 


Wizualizacja obrazów sonarowych 

**/notebooks/test/key_points.ipynb**

W tym pliku znajduje się wizualizacja sekwencji obrazów sonarowych, z zaznaczonymi punktami które są wybierane do śledzenia. 
Na koniec pod nagłówkiem: Carthesian vs Polar coords system visualisation jest komórka która uruchamia porównanie jak zmienia się obraz w polarnym układzie współrzędnych oraz w kartezjańskim układzie współrzędnych. 



Test Bundle Adjustment: 

**/notebooks/test/BA_test.ipynb**

Rzeczywistwa trajektora jest zakłócana o znaną wartość i na podstawie tego wyliczane są idealne wartości poprawek do przepływu optycznego, które w rzeczywistych warunkach powinna zwrócić sieć. Na podstawie tych danych testowany jest moduł bundle adjustment który ma wprowadzać poprawki do pozycji oraz kąta elewacji wiązki dla obserwowanych punktów. 

W pliku używam dwóch implementacji. Pierwszą importuje z pliku bundle_adjustment_v2.py. To jest moja implementacja oparta o optymalizator I rzędu i nie radzi sobie za bardzo z takim zadaniem. Druga, z bundle_adjustment_v3.py to jest w formie eksperymentu i przetestowania innych rozwiązań, ale zaznaczam że jest pisana przez Gemini. 

