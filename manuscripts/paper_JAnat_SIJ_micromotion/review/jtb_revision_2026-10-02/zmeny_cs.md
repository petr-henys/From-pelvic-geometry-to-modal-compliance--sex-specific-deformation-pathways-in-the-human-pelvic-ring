# Revize rukopisu pro otázku anatomické predikce SIJ

Rukopis byl přestavěn kolem otázky, kolik populační variability pohybu SIJ lze
předpovědět z velikosti a několika anatomických znaků a jakou informaci potřebujeme
při změně rozložení zatížení. Změny zahrnují nový název, abstrakt, úvod, hlavní
výsledky, diskusi, závěr, dvě metodické podkapitoly a doplněný supplement.

## Vymezení originality

Na základě potvrzení autora text výslovně uvádí, že nynějších 278 pánví tvoří
podmnožinu souboru 281 pánví v Henyš & Hammer (2025), DOI 10.1111/joa.14160.
Přínos proto není prezentován jako nový nebo nezávislý CT soubor. Nová otázka
spočívá v predikci relativní kinematiky, v měření přínosu omezeného anatomického
popisu na zadržených subjektech a v kontrolovaném testu přenosu skalární odezvy
mezi místy aplikace sil. Doplněná práce Dumont et al. (2009, JTB) zasazuje studii
do teoretického kontextu srovnávací FE analýzy biologických struktur.

## Skutečně doplněné výpočty

**Predikce:** čtyři modely × pět zatížení × dva výsledky, deset opakování
sex-stratifikované desetidílné validace po subjektech. Všechny modely používají
stejné dělení. Model velikosti zahrnuje věk a log objemu; triáda přidává inlet AP,
interspinózní šířku a subpubický úhel. Další modely přidávají pohlaví a jeho
interakci s objemem, nebo používají všech osm dostupných rozměrů. Standardizace,
regrese a přepočet z logaritmické na původní škálu využívají pouze trénovací data.

Triáda snížila RMSE translace o 8,2–31,3 % v pěti zatíženích; všech pět bodových
intervalů přínosu je kladných. Predikční R² translace je 0,309–0,717. Přínos pro
rotaci je 1,2–25,0 %; u LAB2 interval zahrnuje nulu. Devět z deseti intervalů
přínosu triády je kladných. Žádný interval přínosu dvou rozšířených modelů není
celý kladný; 19 z 20 zahrnuje nulu a poslední mírně favorizuje triádu oproti
rozšíření o pohlaví pro rotaci při SP1leg. Podrobnosti jsou v hlavní tabulce 3,
obrázku 3 a supplementu S10.

**Mechanistický test:** pro každého ze 278 subjektů se stejná transverzální síla
400 N na každé straně rozdělila mezi vnitřní prstenec a ischiální aplikační místa
v pěti podílech. Při pevném lineárním systému odpovídají smíšená posunová pole
přesné superpozici. Váhy mají součet jedna, takže společné předpětí je zachováno
jednou. Kinematika byla ze smíšených polí znovu vypočtena, nikoli interpolována
z koncových velikostí pohybu.

Při rozdělení 50:50 byl medián redukce translace vůči skalární kombinaci 21,9 %
(95% interval 20,0–23,7 %). RMSE skalární kombinace činilo 0,0580 mm, zatímco
kombinace podepsaných vektorů 0,00021 mm. Test kvantifikuje ztrátu směrové
informace jako mez přenosu skalární odezvy. Výsledky jsou v hlavní tabulce 5 a
obrázku 6.

Doplněny byly přímé párové kontrasty LAB2–LAB1, krajní chyby standardizace hustoty
a srovnání objemových koeficientů s triádou a bez ní (S11–S13). Podmíněné objemové
koeficienty a heuristické rozměrové exponenty se přesunuly z hlavního argumentu
do supplementu S9/S13 a obrázku S3.

## Vymezení závěrů

Validace je interní a triáda byla vybrána retrospektivně; její původní výběr není
vnořen do validace. Bootstrapové intervaly jsou bodové, podmíněné uloženými
predikcemi a nezahrnují nejistotu nového fitu nebo výběru znaků. Výsledky se týkají
nových subjektů v reprezentované populaci při každém stanoveném zatížení, nikoli
nezávislého CT souboru nebo libovolného nového zatížení.

Mechanistický test používá individuální FE koncové vektory. Není prediktorem
pohybu pouze z anatomie a neurčuje příčinu zbytkových chyb triády mezi subjekty.
Ověřuje odlišnou otázku: které informace potřebujeme k kombinování odezvy při
změně aplikace sil. Superpozice a identita normy nejsou prezentovány jako nová
matematika; biologický přínos spočívá v jejich cíleném testu a kvantifikaci na SIJ.

Závěry jsou omezeny na pasivní statický model se sdílenými měkkotkáňovými
parametry. Simulované pohlavní rozdíly mohou vznikat jen přes reprezentované
individuální vstupy. Zeslabení koeficientu není důkazem mediace nebo ekvivalence.
Standardizace hustoty zachovává úplnou geometrii včetně velikosti a její nízké
průměrné chyby nevylučují větší individuální odchylky.

Tato verze poskytuje podstatně konkrétnější teoretický argument pro JTB než
samotný popis populační funkční anatomie. Nejdůležitější další krok je nezávislá
anatomická a experimentální kinematická validace a predikce směru z anatomických
znaků. Jejich absence je v rukopisu otevřeně uvedena.

## Ověření a soubory

- Nezávislé přefitování všech 4 000 validovaných regresí pomocí statsmodels
  v původních jednotkách reprodukovalo predikce s maximální odchylkou
  2,05 × 10⁻¹³.
- Oba krajní body mechanistického testu souhlasí s původními komponentami rotace
  a translace všech 278 subjektů; maximální komponentová odchylka je
  6,93 × 10⁻¹² při tolerancích 10⁻⁸.
- Ověřeno párování dělení, publikované metriky, definice skalární/vektorové
  kombinace, trojúhelníková nerovnost a souhrnné chyby směšování.
- Podpůrné přefitování reprodukovalo původní objemové a pohlavní koeficienty
  i souhrny geometricko-hustotních variant.
- Vytvořeny hlavní PDF a DOCX a doplňkové PDF a DOCX. Word obsahuje vložené
  obrázky, editovatelné tabulky a nativní matematiku.

Původní texty jsou zálohovány v `before/`. Výpočty a protokoly jsou v
`tables/prediction/`, nové skripty v `analysis/`, dvě nové sady obrázků v
`figures/`. Aktuální postup reprodukce je na začátku `analysis/REPRODUCE.md`.
Historický generátor výsledkového textu nemá přepsat nové ručně udržované sekce.
