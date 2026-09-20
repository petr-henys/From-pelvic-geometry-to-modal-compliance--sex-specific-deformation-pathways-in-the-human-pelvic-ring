# Vědecká revize článku

## Hlavní závěr

Původní verze nebyla připravena k odeslání. Obsahovala chybu referenční konfigurace, která zaměňovala část anatomické variability za mechanický pohyb, nesoulad statistických modelů mezi textem a obrázky a doložené chybné bibliografické záznamy. Revize opravuje výpočet a znovu odvozuje výsledky z uložených FE posunů. Nejde jen o jazykovou korekturu.

## Zásadní oprava výpočtu

`simulation/simulation_loads.py` původně porovnával šablonu X s X + φ + u. Správná dvojice pro pohyb téhož jedince je X + φ a X + φ + u. Nenulová φ tedy dříve vytvářela zdánlivý pohyb i při u = 0. Afinní korekce sakra obecné neafinní rozdíly anatomie neodstraní.

To obzvláště ohrožovalo interpretaci „dominance geometrie“: full a shape-only sdílely stejnou anatomickou složku chyby, material-only nikoli. Původní numerické závěry nelze obhájit pouhým přejmenováním metrik.

Oprava využívá uložené posuny, původní parametry interpolace a archivované souřadnice FE uzlů. Pořadí uzlů je kontrolováno proti všem pěti referenčním polím. Párování subjektů vychází ze společných indexovaných vstupních polí a pořadí průchodu v kódu; pomocné archivy nemají nezávislý manifest ID, což zůstává omezením provenance. Původní FE archivy nejsou přepisovány. Nové výsledky jsou v `tables/corrected`, původní znění a statistické tabulky v `review/before_revision`.

Kontrolní skript reprodukoval původní SP2leg výsledky u šesti rovnoměrně rozložených indexů s tolerancí 1e-8. Nejde tedy pouze o podezření ze čtení kódu. Nulový test opravené reference probíhá pro každého jedince.

## Matematika a implementace

- Povrchová penalizace γ má jednotku N/mm³, nikoli N/mm. γu je trakce.
- Slabá forma nyní obsahuje také tuhost vazů a pravou stranu od předpětí.
- Archiv skutečně používá E = 1 MPa pro SIJ i symfýzu. Údaje 15 a 5 MPa v příloze byly nesprávné.
- Vazy jsou linearizované pružiny s geometrickou tuhostí od předpětí; solver neřeší nelineární přepínání „pouze tah“.
- Extrakční rovnice úhlů odpovídají Rz Ry Rx, tedy extrinzické sekvenci xyz. Původní příloha měla opačné pořadí. Opravena byla i singularitní větev kódu; rekonstrukční test zahrnuje obě znaménka ±90°.
- Norma Cardanových úhlů je přibližná malouhlová metrika, nikoli přesný invariantní úhel rotace.
- PCA „symmetry plane“ není optimalizace zrcadlové symetrie. AP/CC znaménka nejsou nezávisle anatomicky zakotvena.
- Afinní korekce odstraňuje i deformaci sakra, nejen rigidní drift. Doplněna citlivost na rigidní korekci u šesti jedinců × pěti zátěží: maximální změna translace 0,00553 mm a rotace 0,02290°. Není to validace celé populace.
- Absolutní ML složka není otevření kloubní štěrbiny; PCA osy nejsou normály kloubních ploch.
- Shape-only používá prostorově proměnnou bodovou mediánovou hustotu, nikoli homogenní kost.
- Kostní konstitutivní zákon má konstantní větev pod prahem hustoty. Výsledek proto netestuje libovolný vliv osteoporózy či mineralizace.
- Archivovaný objem je geometrický objem tří kostních těles, nikoli objem porodních cest a nikoli jen kortikální tkáně.

## Statistika a obrázky

Původní text M2 neobsahoval sex×volume, ale tabulka a lesní graf ano. Nyní se používá jednotný M2 bez této interakce; interakce zůstává výslovně ve sdružených a allometrických modelech. Všechny číselné pasáže výsledků a abstraktu generuje skript z opravených tabulek.

Původní allometrický obrázek kreslil neadjustované přímky s popisky adjustovaných exponentů. Křivky nyní vycházejí ze stejného modelu; hlavní obrázek navíc přehledně zobrazuje exponenty s intervaly. Původní rezidualizace výsledku pouze podle kovariát nemohla přímo zobrazit adjustovaný rozdíl pohlaví; nahrazena skutečnými koeficienty M0–M2. Stupně a milimetry mají oddělené osy. Popisky označují korigované hodnoty jako q, nikoli p.

Poměry rozptylů nejsou aditivní rozklad příčin ani „vysvětlená variabilita“. Mohou přesahovat 100 %. Doplněny párové chyby a identity-line R². Odstraněno „partial R²“ odvozované z robustního Waldova statistika, které není klasickým podílem vysvětleného rozptylu.

Primární test složkových mediánových kontrastů byl změněn na exaktní znaménkový test. U asymetrických rozdílů totiž signed-rank test vedl k významnosti i při intervalu mediánu přes nulu. Signed-rank výsledky zůstávají jako diagnostika v CSV; hlavní test nyní odpovídá interpretovanému mediánu.

FDR rodiny jsou explicitní: 12 složkových kontrastů; 3 stojné kontrasty; 6 M2 pohlavních koeficientů; 20 allometrických sklonů; 10 allometrických interakcí. Nekorigované korelace jsou označeny jako exploratorní. Nevýznamná interakce není důkaz stejných sklonů. Log(scale) je přesná reparametrizace log(volume), nikoli nezávislá kontrola robustnosti.

Hypotézy jsou přepsány do měřitelných kontrastů a označeny jako retrospektivní. LAB1–LAB3 nejsou skutečné fáze porodu. Podle souřadnic archivované šablony míří obě ML dvojice navenek; původní označení LAB1 jako komprese bylo v rozporu se silami. Jejich účinek na konkrétní kloub však nelze vyvodit jen ze směru aplikovaných sil.

## Originalita

Samotný počet modelů ani zjištění, že geometrie ovlivňuje SIJ, nejsou originálním příspěvkem. Stejní autoři již publikovali [studii 281 FE pánví v Journal of Anatomy](https://doi.org/10.1111/joa.14160). Obhajitelný přínos zde představuje kombinace párových kontrastů pěti zátěží, explicitních složek pohybu a porovnání geometrických a materiálových variant.

Nutno doložit překryv zobrazovacích dat s předchozí prací. Z podobnosti počtu subjektů nelze překryv automaticky určit, ale nelze ani tvrdit nezávislou populační validaci. Novost článku má být formulována jako nová analýza, nikoli automaticky nová kohorta.

[Heyland a kol. (2025)](https://doi.org/10.1007/s11517-025-03396-w) ukazují význam předpětí vazů. Jejich dvě reprezentativní geometrie pocházely z většího zobrazovacího souboru; není správné zaměňovat tento soubor za 818 individuálních FE modelů. Práce podporuje potřebu materiálové citlivosti, nikoli tvrzení, že zdejší kostní density-only poměr omezuje vliv všech měkkých tkání.

## Realističnost a užitečnost

Výsledky lze použít jako interní charakteristiku modelové kohorty a k návrhu experimentů nebo výběru zatěžovacích konfigurací. Bez srovnatelných experimentů nelze označit absolutní hodnoty za fyziologické normy. Chybí dynamika, svaly, kontakt plodu, těhotenské tkáňové změny a výsledky porodů či bolesti. Věkové rozpětí 16–91 let nereprezentuje populaci rodiček.

Odstraněna byla tvrzení o prokázané samosvornosti, zachování kongruence, zvětšení porodních cest, příčině bolesti, doporučení fixace a vysvětlení evolučního porodnického dilematu. Přímý klinický přínos z těchto dat zatím doložen není. Vědecká užitečnost spočívá v reprodukovatelných podmíněných mechanických srovnáních.

Pro silnější článek je třeba přidat validaci při odpovídajícím zatížení, konvergenci sítě, kontrolu Jacobianů, nejistotu chrupavky a vazů a případně přímé změny rozměrů porodních cest. To jsou chybějící experimentální či simulační důkazy, nikoli problémy, které lze opravit rétorikou.

## Bibliografie

Kontrola metadat DOI prokázala čtyři nesouvisející citace, které byly odstraněny:

| Původní klíč | DOI ve starém souboru | Skutečný obsah podle metadat |
|---|---|---|
| Keizer2019 | 10.1016/j.jbiomech.2019.05.039 | Variabilita chůze u Parkinsonovy nemoci |
| Meijer2014 | 10.1016/j.spinee.2013.10.041 | Operace bederní páteře při chronické bolesti |
| Yang2020 | 10.1115/1.4046595 | Technický mechanismus s kvazinulovou tuhostí |
| BiomechPregnancy2022 | 10.1016/j.whi.2022.03.004 | Kvalitativní studie zvažování dalšího těhotenství po předčasném porodu |

Raw metadata jsou v `bibliography_crossref.json`. U části původních záznamů Crossref vracel HTTP 429; to není důkaz jejich neexistence. Revidovaný text používá zúženou bibliografii relevantních zdrojů. Doplněna chybějící předchozí práce autorů a aktualizováni někteří neúplní autoři záznamů.

## Zbývající autorské kroky

Ověřit původ a etické schválení konkrétní kohorty, vysvětlit vztah k dřívějším publikacím, dodat skutečný archivní odkaz, zkontrolovat institucionální údaje a schválit novou verzi oběma autory. Přístupnost anonymního úložiště ani souhlas všech autorů nebyly předstírány. Podrobnosti jsou v `AUTHOR_CHECK_REQUIRED.md`.

## Výsledek dokončeného přepočtu

Celkem 278 subjektů × 5 zátěží × 3 varianty = 4 170 záznamů, bez chybějících hodnot. Původní reference při nulovém posunu vytvářela medián zdánlivé rotace 1,425° a translace 0,624 mm.

| Výsledek | Opravená hodnota / interpretace |
|---|---|
| SP2leg translace | 1,058 mm |
| SP1leg translace | 1,627 mm |
| LAB1 / LAB2 / LAB3 translace | 0,154 / 0,255 / 0,574 mm |
| SP2leg / SP1leg asymetrie | 0,065 / 1,602 mm |
| Adjustovaný ženský rozdíl LAB1 / LAB2 / LAB3 | +0,013 / −0,031 / +0,002 mm; žádný významný po FDR |
| H1 | Párové kontrasty podpořeny |
| H2 | Celý navržený vzor LAB1/LAB2 nepodpořen |
| Shape-only/full rozptyl | 96,43–120,61 %, nikoli aditivní vysvětlené podíly |
| Material-only/full rozptyl | až 5,789 %, nikoli původně tvrzených <0,32 % |
| Allometrie | Všech 20 podmíněných sklonů záporných po FDR; nikoli pouze jednostranný stoj |

Pooled korelace úhlu a LAB1 translace je +0,444, ale uvnitř mužů −0,063 a žen −0,145. Původní mechanický výklad pozitivní korelace proto nelze zachovat.

Kontroly: čtyři přenositelné regresní testy prošly; nezávisle přepočteno 12 znaménkových testů, ověřena shoda M2 mezi tabulkami a modely a přesná trojnásobná transformace exponentů volume→scale. Širší FE testy nebylo možné sesbírat v tomto prostředí kvůli chybějícímu mpi4py. Postprocessing nepouští nové FE simulace.
