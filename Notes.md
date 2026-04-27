Walidacja na tkaije samje liczbie update, jak trening. Sprawdzić czy ta ilośc wystarczy 


Oto szczegółowa notatka dotycząca zmian w Twoim systemie, oparta na analizie kodu oraz artykułu naukowego "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.).

Notatka: Implementacja wag ufności (Homoscedastic Uncertainty)Aby wyeliminować problem "kolapsu wag" (uciekania wag do zera) i umożliwić stabilny trening odometrii, należy wprowadzić następujące zmiany:


1. Plik: update.py

Zmiana: 
Usunięcie warstwy nn.Sigmoid() z modułu self.w.

Dlaczego: 
Artykuł wskazuje, że sieć powinna przewidywać logarytm wariancji ($s := \log \sigma^2$), który jest wartością nieograniczoną ("unconstrained scalar values").

Uzasadnienie techniczne: 
Warstwa Sigmoid blokuje wartości w przedziale (0, 1). Gdy sieć uczy się ignorować błędy, spycha wynik w stronę zera, gdzie gradient funkcji Sigmoid zanika (Vanishing Gradient). Usunięcie jej pozwala optymalizatorowi swobodnie korygować poziom ufności poprzez operowanie na logarytmie.


2. Plik: bundle_adjustment.py

Zmiana:
Sposób aplikowania wag w funkcji forward. Zamiast project_err * weights, należy użyć project_err * torch.exp(-self.weights).

Dlaczego: 
Zgodnie z artykułem, w modelu regresyjnym waga błędu to $1/\sigma^2$. Ponieważ sieć teraz wypluwa $s = \log \sigma^2$, to waga $\frac{1}{\sigma^2}$ jest matematycznie równoważna $\exp(-s)$.

Uzasadnienie techniczne: 
Mapowanie wykładnicze (exp) zapewnia, że waga błędu w optymalizatorze Bundle Adjustment zawsze będzie dodatnia, bez konieczności stosowania sztucznych ograniczeń czy epsilonów.

3. Plik: DPSO_LightningModule (Moduł treningowy w New Text Document.txt)

Zmiana: 
Modyfikacja funkcji straty loss_weighted w krokach training_step i validation_step.Nowa formuła: loss_weighted = torch.exp(-weights) * err_raw + weights (lub + 0.5 * weights dla pełnej zgodności z log-likelihood).

Dlaczego: 
W Twoim obecnym kodzie kara za małe wagi to -loss_w_weights * torch.log(weights + 1e-6). Artykuł proponuje prostszy i stabilniejszy człon regularyzacyjny: + log σ.

Uzasadnienie techniczne: 
Jak opisano w sekcji 3.2 artykułu, ostatni człon funkcji straty działa jako regularyzator. Zapobiega on nieskończonemu zwiększaniu "szumu" (czyli zmniejszaniu wag), co w Twoim poprzednim treningu pozwalało sieci "oszukiwać" i ignorować dane.

Podsumowanie teoretyczne (z artykułu):
Przejście na przewidywanie logarytmu wariancji ($s$) zamiast bezpośrednich wag ($w$) poprawia stabilność numeryczną, ponieważ unika dzielenia przez zero w funkcji straty. Artykuł pokazuje, że taka dynamiczna adaptacja wag pozwala modelowi automatycznie ustalić relację między różnymi składowymi błędu (np. błędem azymutu i odległości), co w Twoim przypadku prowadzi do dokładniejszej trajektorii.

____



2. "Bomba" NaN w funkcjach trygonometrycznych (Eksplozja Gradientów)
Model uczy się kwaternionów poprzez obliczenie odległości kątowej. W Twojej implementacji to tykająca bomba, która wyzeruje całą sieć jednym błędem NaN (Not a Number) podczas treningu.

Gdzie: utils.py, funkcja approx_movement.

Błąd: Masz kod:

Python
w = torch.clamp(q_diff[:, -1:], min=-1.0, max=1.0)
q_diff_angle = 2 * torch.arccos(w)
Dlaczego to niszczy sieć: Matematyczna pochodna funkcji arccos(x) to -1 / sqrt(1 - x^2). Jeśli w wynosi dokładnie 1.0 (lub -1.0), mianownik jest zerem, a gradient dąży do nieskończoności! Gdy model w jednej klatce zgadnie idealnie brak ruchu (tożsamościowy kwaternion, w=1.0), wsteczna propagacja wyrzuci NaN na wszystkie wagi Twojej sieci GRU i CNN, natychmiast je niszcząc.

Rozwiązanie: Ogranicz clamp przed wejściem w arccos używając lekkiego marginesu (epsilon):

Python
w = torch.clamp(q_diff[:, -1:], min=-1.0 + 1e-6, max=1.0 - 1e-6)
3. Crash przy treningu (Batch Size > 1) – Błąd Broadcasting'u
W notatniku test_lightning.ipynb testowałeś sieć na pozycjach z batch_size = 1. Kiedy jednak odpalisz pełny trening np. z batch_size = 4, Twój kod ulegnie całkowitej awarii z powodu błędu wymiarowości.

Gdzie: utils.py, funkcja approx_movement.

Błąd: translation_diff = (translation2 - translation1) / dt12 * dt23.

Dlaczego to niszczy sieć: translation2 - translation1 to tensor o kształcie (Batch, 3). Zmienne dt12 i dt23 (jako różnice czasu ze zmiennej timestamp) mają kształt jednowymiarowy (Batch,). W PyTorchu podzielenie (B, 3) przez (B,) wywoła RuntimeError: The size of tensor a (3) must match the size of tensor b (B).... Działało Ci to na teście tylko dlatego, że dla (1, 3) / (1,) PyTorch może awaryjnie naciągnąć wymiar 1.

Rozwiązanie: Rozszerz wymiary czasu, żeby stały się kolumnowe:

Python
dt12 = (t2 - t1).unsqueeze(-1)
dt23 = (t3 - t2).unsqueeze(-1)
translation_diff = (translation2 - translation1) / dt12 * dt23
4. Uszkodzony Bundle Adjustment dla Batches
To rozwinięcie problemu, o którym wspominałem na samym początku naszej konwersacji, jednak wymaga ono gruntownej przebudowy, by trening mógł zadziałać poprawnie.

Gdzie: bundle_adjustment.py.

Błąd: Łączysz wymiary Batch i Poses przed utworzeniem zmiennych do optymalizacji:
init_poses = init_poses.view(1, poses_n, 7)
a następnie wycinasz stałe pozy:
self.poses_anchor = init_poses_se3[:, :freeze_poses, :].

Dlaczego to niszczy sieć: Załóżmy, że trenujesz 4 sekwencje po 10 klatek (b=4, act_n=10, poses_n=40). Twoje freeze_poses to 2. Powyższy kod zamrozi absolutnie dwie pierwsze klatki z całego 40-elementowego wektora, co oznacza, że zamrozisz układ odniesienia tylko dla pierwszego elementu w batchu! Sekwencje 2, 3 i 4 nie będą miały zakotwiczenia – będą floatować w przestrzeni, a ich Loss z Bundle Adjustment przekaże kompletnie chaotyczny sygnał do sieci.

Rozwiązanie: Rozbij inicjalizację na (b, act_n, 7), stwórz parametry, a spłaszczaj dopiero przed rzutowaniem punktów.

Zamień sekcję __init__ na to:

Python
init_poses = init_poses.view(self.b, self.act_n, 7)
# self.poses_anchor zachowuje teraz zamrożone pozy dla KAŻDEGO przykładu z batcha
self.poses_anchor = pp.SE3(init_poses[:, :freeze_poses, :])

if freeze_poses >= self.act_n:
    self.split_poses = False
else:
    self.translation_optim = nn.Parameter(init_poses[:, freeze_poses:, :3])
    self.rotation_optim = nn.Parameter(init_poses[:, freeze_poses:, 3:])
    self.split_poses = True
W funkcji forward() w BA scalaj to z powrotem i spłaszczaj:

Python
if self.split_poses:
    rotation_optim_norm = F.normalize(self.rotation_optim, p=2, dim=-1)
    poses_optim = torch.cat([self.translation_optim, rotation_optim_norm], dim=-1)
    poses_raw = torch.cat([self.poses_anchor.tensor(), poses_optim], dim=1)

    # Spłaszczenie do 1D tylko na potrzeby globalnego adresowania po krawędziach
    poses_raw = poses_raw.view(1, self.b * self.act_n, 7) 
    poses = pp.SE3(poses_raw)
Wdrożenie tych poprawek w połączeniu z wagami Kendalla da Twojej sieci wreszcie stabilne i poprawne środowisko do nauki (brak NaN, poprawne osie X/Y w sieciach splotowych, odseparowane batche).
