**Posudek rukopisu „Anatomical scale and architecture organize load-dependent sacroiliac motion“ — 2. října 2026**

**Celkové hodnocení:** Rukopis obsahuje použitelný a potenciálně publikovatelný populační výsledek, ale současnou verzi nedoporučuji odeslat. Nejdůležitější překážkou je doložená lokální inverze geometrického mapování, jejíž mechanický dopad není vyhodnocen. Originalita je především v kombinaci kinematických endpointů, více zatížení a párovaných geometricko-hustotních variant; obecná teze, že anatomie ovlivňuje mechaniku SIJ, nová není. Pro Journal of Theoretical Biology je téma přijatelné, současný biologický a mechanistický přínos však potřebuje přesnější vymezení a silnější podporu.

**Rozsah posouzení a ověření.** Posoudil jsem současné `main.tex`, všechny zahrnuté sekce, PDF sestavené 2. října 2026, Supporting Information, publikované tabulky, příslušné analytické skripty a vybrané části FE implementace. V PDF jsem vizuálně zkontroloval zejména obrázky porovnání variant a škálovacích sklonů. Pro originalitu a zaměření časopisu jsem provedl cílenou rešerši primárních publikací a oficiálního popisu vydavatele. Nejde o vyčerpávající bibliometrickou rešerši ani kontrolu plagiátorství.

Z uložených individuálních endpointů jsem nezávisle přepočítal deset publikovaných HC3 log–log modelů, osmnáct vnořených modelů pohlavních kontrastů, variantové variance a shodu, mediány asymetrie a citované Spearmanovy korelace. Kontrolu Jacobianů jsem znovu provedl ze zdrojových geometrií a konektivity; použitý kontrolní skript navíc ověřuje nejhorší element každého subjektu přímou RBF interpolací a druhým výpočtem `det(I + grad(phi))`. Neprovedl jsem nové FE řešení, opravu geometrií ani experimentální validaci. Přidané analýzy jsou diagnostikou posudku; nemění primární výsledky nebo rukopis.

**1. Numerická správnost: doložená chyba geometrického mapování je zásadní.**

V `sections/methods.tex:26` a `sections/appendix.tex:14` se uvádí bijektivní anatomická registrace a její RBF reprezentace na FE síti. Bijektivita zdrojové registrace však sama nezaručuje přípustnost následné diskrétní interpolace. Přepočet všech 278 geometrií potvrdil:

| Kontrola | Výsledek |
| --- | ---: |
| Počet tetraedrů na geometrii | 166 808 |
| Počet kontrolovaných dvojic subjekt–element | 46 372 624 |
| Subjekty s alespoň jedním záporným Jacobianem | 224 z 278, přibližně 80,6 % |
| Záporné Jacobiany celkem | 425 |
| Nejnižší Jacobian | −2,5834614973 |
| Nejvyšší počet převrácených elementů u subjektu | 7 |
| Maximální podíl referenčního objemu v dotčených elementech | 0,000527306 % |

Čísla odpovídají předchozí uložené diagnostice v `tables/revision/mapping_summary.json`. Současný rukopis ani jeho supplement tento nález neuvádějí. Interní `review/submission_revision.md` eviduje, že příslušný odstavec byl později odstraněn a že mechanický dopad nebyl přepočítán.

Proč to vadí: `simulation/solver_base.py:123` násobí elastickou bilineární formu znaménkovým `J`; také povrchový Jacobian používá toto znaménko. V převráceném elementu tak nevzniká fyzikálně přípustný příspěvek integrace na orientaci zachovávající subjektové geometrii. Malý objem není důkazem malého vlivu: význam závisí na poloze, deformaci, lokálním zkreslení a vztahu ke kloubům či zatížení. Přímý LU solver může vrátit řešení i pro problematickou matici; úspěšná faktorizace neověřuje fyzikální přípustnost.

To **nedokazuje, že všechny výsledky jsou chybné**, ale brání tomu, abych jejich FE původ označil za numericky spolehlivý. Full a G sdílejí stejné geometrické mapy, takže jejich vzájemná shoda tento problém neověřuje. D používá template s `J = 1`; porovnání variant tak navíc není vystaveno geometrickému defektu symetricky.

Před odesláním je potřeba doložit přípustné mapování, znovu vyřešit dotčené modely a ukázat dopad opravy na hlavní endpointy a závěry. Prosté nahrazení `J` za `abs(J)` neřeší samotnou lokální inverzi či možné překrytí geometrie. Pokud by byla zvolena jiná metoda nápravy, musí mít vlastní geometrické a mechanické zdůvodnění. Uvedení vady v limitacích by zlepšilo transparentnost, ale samo nezajistí správnost výsledků.

**2. Originalita: existuje samostatný přínos, ale chybí nejbližší předchozí práce.**

Nejvýznamnější chybějící citací je práce stejných autorů [Henyš a Hammer, 2025, *Sacroiliac joint auricular surface morphology modulates its mechanical environment*](https://doi.org/10.1111/joa.14160). Již obsahuje 281 CT odvozených FE modelů, populační morfomechanické analýzy SIJ a vztahy velikosti kloubních struktur ke stresu a deformaci. V nynější bibliografii `bib/refs.bib` ani v textu tato práce není.

Nový rukopis má odlišné těžiště: relativní kinematiku SIJ, pět zatěžovacích konfigurací, změnu pohlavních koeficientů po anatomické adjustaci a srovnání full/G/D. Tato kombinace může založit samostatnou publikaci. Samotný rozsah souboru však nelze představovat jako nový průlom v populačním modelování SIJ.

Je nutné výslovně uvést, zda a nakolik se překrývají CT kohorty, jak se liší model, zatížení a endpointy a co nový článek zjišťuje nad rámec staršího. Překryv konkrétních subjektů jsem nezjistil; nelze jej odvodit pouze z podobného počtu modelů. Pokud jde o navazující analýzu stejného souboru, je to přijatelné, ale musí to být jasné. Citace BoneDat tento vztah nenahrazuje.

Za nejsilnější kandidáty na originalitu považuji kvantifikovanou přesnost hustotního zjednodušení pro konkrétní kinematické endpointy a propojení populačních pohlavních kontrastů s anatomickými kovariátami při různém zatížení. Unilaterální asymetrie a obecný vliv velikosti mají menší konceptuální originalitu. Z článku zatím nevychází nový obecný škálovací zákon.

**3. Jsou výsledky a závěry konzistentní? Převážně ano, pokud je čteme jako podmíněné výsledky modelu.**

| Tvrzení | Co ukazují výsledky | Hodnocení argumentace |
| --- | --- | --- |
| Unilaterální podpora zvětšuje pohyb a asymetrii | Párované změny +1,048°; +0,582 mm; +1,509 mm asymetrie | Dobře podpořeno v modelu; nezměněná celková síla neznamená nezměněné momenty |
| Obě transverzální dvojice zvětšují ML složku a zmenšují CC složku | CC klesá u obou; ML roste u LAB2, u LAB1 interval zahrnuje nulu | H2 je pouze částečně podpořena; rukopis to správně přiznává |
| Anatomická adjustace tlumí pohlavní rozdíly | LAB translace M0→M2: +0,055→+0,013; +0,071→−0,031; +0,258→+0,002 mm | Podpořeno jako změna regresního kontrastu; nezakládá mediaci ani ekvivalenci |
| G zachovává odpověď full | Translace RMSE nejvýše 0,027968 mm; minimální identity-line R² 0,945707 | Dobře doložena shoda těchto dvou realizací modelu; obecnější fyziologický závěr vyžaduje další ověření |
| Objem má záporný podmíněný vztah k pohybu | Všech 20 adjustovaných sklonů je záporných; max q = 0,00187363 | Statisticky podpořeno v dané specifikaci; není to přímý test uniformního škálování |
| SIJ pohyb vysvětluje rozšíření porodních cest | Signed gapping ani porodní kanál nejsou měřeny | Takový závěr by podporu neměl; aktuální text jej správně nečiní |

Zkontrolované publikované sklony a interakční p-hodnoty byly reprodukovány s maximální absolutní odchylkou přibližně `3 × 10⁻¹⁴`; koeficienty a intervaly vnořených modelů s odchylkou pod `7 × 10⁻¹⁶`. Přepočet variance/agreement souhrnů se lišil nejvýše přibližně `1,4 × 10⁻¹⁴`. Mediány asymetrie 0,064937 a 1,601572 mm i všechny tři citované korelace odpovídají textu. U těchto tvrzení jsem nenašel aritmetický nesoulad.

Důležité je oddělit tuto velmi dobrou statistickou reprodukovatelnost od správnosti FE vstupních řešení. Reprodukovatelná regrese sama neověřuje mechanický model.

**4. Škálování: správná dimenzionální úvaha, ale jiný statistický estimand.**

Pro geometricky podobnou lineárně elastickou strukturu při konstantní síle a modulu je úvaha `u ∝ F/(EL)` a `θ ∝ F/(EL²)` správná. Při `V ∝ L³` dává referenční sklony −1/3 a −2/3. Současný článek však neprovádí sérii homoteticky škálovaných pánví. Regrese mění objem při konstantních AP a interspinózní vzdálenosti v milimetrech a konstantním subpubickém úhlu. To je odlišná cesta anatomickým prostorem.

Text i popisek Figure 7 tento rozdíl poctivě uvádějí. Přesto grafické reference a opakovaná zmínka o numerické blízkosti ve výsledcích, abstraktu a diskusi mohou dát této shodě větší váhu, než má. Proximity ke dvěma referenčním hodnotám sama neověřuje podobnost geometrií ani biologickou allometrii.

Nová diagnostika posudku ukazuje, jak velký vliv má přidání triády. V obou specifikacích zůstává věk, pohlaví a interakce objem×pohlaví; mění se pouze přítomnost triády:

| Endpoint a skupina | Sklon bez triády | Publikovaný sklon s triádou |
| --- | ---: | ---: |
| SP2leg, translace, muži | −0,231 | −0,366 |
| LAB1, translace, ženy | −0,481 | −1,003 |
| LAB2, translace, muži | −0,382 | −0,818 |
| LAB3, translace, ženy | −0,382 | −0,904 |

Všech 20 bodových sklonů zůstává záporných i bez triády, ale už ne všech 20 testů přežije stejnou BH korekci; nejvyšší q je 0,2912. Nejde o důkaz chybnosti adjustovaného modelu. Dokládá to, že konkrétní velikosti a společná statistická významnost závisejí na definici podmíněného srovnání. Interpretace velmi strmých LAB sklonů musí zahrnout tuto skutečnost.

Také samotný FE model nesplňuje čistou geometrickou podobnost: konstantní ligamentová tuhost v N/mm mění poměr pružinové a kontinuální tuhosti s velikostí, pretension přispívá geometrickou tuhostí a nominální tkáně nemají doloženou fyziologickou velikostní závislost. Nelze proto automaticky očekávat referenční sklony u současných simulací.

Pro skutečný mechanistický příspěvek doporučuji kontrolovanou sérii škálovaných geometrií s výslovně určeným škálováním vazů a pretension, následovanou analýzou odchylek populačních geometrií. Alternativou jsou velikostně normalizované geometrie nebo vhodně definované bezrozměrné tvarové proměnné; ty ale samy nejsou důkazem kauzality. Lze zkoumat normalizovanou odezvu `uEL/F` a `θEL²/F`, pokud je předem smysluplně definován referenční modul a délka pro heterogenní pánev. Pokud tyto analýzy nejsou cílem článku, doporučuji ponechat závěr o podmíněné objemové asociaci a omezit důraz na referenční exponenty.

**5. Pohlaví a architektura: interpretace je zdrženlivá, ale účinek tvaru není izolován.**

Aktuální formulace „adjustment attenuates sex-associated differences“ odpovídá datům a je vhodnější než tvrzení, že pohlaví nemá žádný mechanický význam. Intervaly LAB2 a LAB3 připouštějí rozdíly translace v řádu setin mm; bez předem definované tolerance nelze usuzovat na mechanickou ekvivalenci.

Skupiny se výrazně liší v subpubickém úhlu, interspinózní vzdálenosti a objemu. Průměrný subpubický úhel je 85,80° u žen a 69,11° u mužů. Adjustovaný koeficient proto srovnává anatomicky podmíněné skupiny s omezeným překryvem. VIF 5,55 není extrémní, ale nevylučuje extrapolaci v mnohorozměrném prostoru ani citlivost výběru kovariát. Doporučuji ukázat společnou podporu anatomických proměnných a citlivost na zdůvodněné alternativy triády. HC3 koriguje odhad kovariance, nikoli chybnou funkční specifikaci nebo nedostatečný překryv.

Pooled korelace subpubického úhlu s LAB1 translací `r = 0,444` a slabě záporné korelace uvnitř pohlaví jsou správně interpretovány jako rozdíl mezi pooled a stratifikovanou asociací. Tato demonstrace zároveň oslabuje případné tvrzení o přímém vlivu právě tohoto úhlu. Jeho mechanický význam by vyžadoval kontrolovanou geometrickou perturbaci či jinou identifikaci mechanismu.

Článek definuje „architecture“ jako geometrii a proporce nad rámec skalární velikosti. Varianta G ale zachovává současně velikost i veškerou geometrii. Full/G/D tedy neoddělují **size od shape**. Regresní adjustace tuto experimentální separaci nenahrazuje. Výsledky podporují význam zachování celkové individuální geometrie; zatím nekvantifikují nezávislý přínos architektury při stejné velikosti. To je důležitá mezera mezi širokým názvem a konkrétním důkazem.

**6. Hustota: silný praktický výsledek s úzkou platností.**

Vedle neaditivních poměrů variance jsou identity-line R², RMSE a MAE skutečnou předností analýzy. Samotná shoda variancí by shodu jednotlivců nezaručovala; rukopis zde používá další správné ukazatele. Současné upozornění, že G používá prostorově proměnnou mediánovou hustotu, je také důležité.

Výsledek však platí pro implementovaný zákon s nízkohustotní větví `E = 2398 MPa`, pevné vlastnosti chrupavky a vazů a konkrétní endpointy. Pokud zákon potlačí variabilitu významné části kostních vstupů, je malý vliv standardizace hustoty zčásti očekávatelný. Diskuse tuto možnost uvádí, ale nevypočítává její velikost. Nelze z toho odvodit, že hustota obecně není důležitá pro SIJ nebo pro osteoporózu.

Před mechanistickým zobecněním je potřeba popsat skutečný rozsah variability **modulu**, zmapovat citlivost na použitý zákon a alespoň v reprezentativních modelech ověřit vliv chrupavky, vazů a pretension. Transformace hustoty na modul probíhá před RBF interpolací do elementů (`simulation/data_mapper.py`); podíl zdrojových hustot pod prahem proto není automaticky podílem FE objemu pod prahem. Z tohoto rozdílu není vhodné odvozovat neexistující elementovou klasifikaci.

RMSE ≤0,028 mm také neznamená chybu nejvýše 0,028 mm u každého člověka. Nový přepočet maximální individuální chyby translace dává 0,2195 mm v SP2leg a 0,0861 mm v LAB2; 95. percentily jsou 0,0486 a 0,0229 mm. To nezpochybňuje uváděné RMSE, ale doporučení zjednodušit hustotu by mělo obsahovat rozdělení individuálních chyb a přijatelnou toleranci. „Prioritize geometry“ je obhajitelné doporučení pro tuto implementaci; klinická přijatelnost zjednodušení ověřena není.

**7. Zatížení, pretension a definice endpointu.**

Párovaná srovnání SP1leg/SP2leg jsou dobře navržena pro otázku změny rozdělení síly. Článek správně odděluje velikost výslednice od momentů. H2 má vhodně přiznanou částečnou podporu. Rozdíl významností LAB1 a LAB2 vůči SP2leg sám o sobě netestuje LAB1 proti LAB2; diskuse tuto mezeru výslovně uvádí.

Pro podporu tvrzení o místě aplikace jsem nově provedl přímé párované srovnání LAB2−LAB1. Median rozdílu absolutních komponent je ML +0,0576 mm [0,0536; 0,0634], AP +0,1108 mm [0,0879; 0,1315] a CC −0,0204 mm [−0,0309; −0,0125]. Všechny tři sign testy přežívají BH korekci v této nové třítestové rodině. Jde o exploratorní diagnostiku posudku z existujících endpointů, nikoli o již publikovanou analýzu. Podporuje rozdílné komponentové odpovědi dvou míst aplikace, za předpokladu přípustnosti vstupních FE řešení.

Další omezení je referenční stav: kinematika je měřena od anatomické konfigurace s nulovým elastickým posunem; zatížené řešení obsahuje i pretension. Absolutní endpoint proto není čistým přírůstkem vyvolaným vnější silou. U lineárního modelu by se pretension odečetla při rozdílu **vektorových polí** mezi dvěma zatíženími. Obecně se však neodečte v rozdílu norem `|u_a|−|u_b|` či jiných nelineárních kinematických souhrnů. Z toho plyne potřeba samostatné pretension-only konfigurace, pokud mají být endpointy interpretovány jako pohyb od mechanicky ustáleného klidového stavu. Aktuální text svůj convention správně vysvětluje, ale jeho vliv neověřuje.

Affine sacral correction odstraňuje i strain sacra. Výsledná relativní kinematika je jasně definovaný postprocessingový ukazatel, fyzikálně však není prostým odstraněním rigidního pohybu pozorovatele. Šestisubjektová kontrola dává malé rozdíly a je užitečná; neověřuje celou kohortu ani stabilitu následných regresních sklonů. Pro posílení článku doporučuji kontrolu na celé kohortě nebo alespoň záměrně vybraných extrémech velikosti, geometrie a pohybu, včetně porovnání hlavních inferencí.

Absolutní ML komponenta neznamená signed gapping a Cardan norm není invariantní principal angle. Rukopis obojí správně uvádí. U malých rotací není Cardan norm nejvyšší prioritou oprav; chybějící fyziologická a numerická validace je podstatnější.

**8. Validace a dostupnost: limity jsou přiznány, jejich dopad ale zůstává nevyřešen.**

U současného modelu chybí doložená konvergence klíčových kinematických endpointů, posouzení transferu geometrie a srovnání s experimentem při odpovídajícím zatížení. P1 tetraedry s chrupavkou `ν = 0,45` odůvodňují kontrolu citlivosti na discretizaci; z použitého řádu ale nelze bez výsledků prohlásit, že model skutečně trpí lockingem. Je také vhodné vyčíslit malé deformace v nejvíce namáhaných měkkých tkáních; malé rigidní rotace kostí samy nezaručují malé lokální strains.

Citovaná experimentální práce [Hammer et al., 2019](https://doi.org/10.1111/joa.12924) uvádí pod bodyweight zatížením dominantní translaci přibližně 0,32 mm a rotaci 0,16°. Zde je SP2leg medián normy translace 1,058 mm a rotace 0,956°. Nejde o přímo shodné endpointy, síly ani boundary conditions, a proto toto porovnání nedokazuje chybu modelu. Vyžaduje však vysvětlení rozdílů a vhodnější validaci než pouhé konstatování, že jde o malé pohyby.

Data Availability je transparentní: BoneDat je zdrojem anatomie, nikoli úložištěm nových FE výsledků a kódu. Pro přezkoumatelnost této práce doporučuji zveřejnit anonymizované odvozené tabulky, identifikační manifesty pairing, konfigurace a skripty. Uvedené párování podle řádků sdílených vstupů je plausibilní, ale nezávislý manifest identifikátorů variant nebyl dostupný. Nejde o nalezenou chybu pairing; jde o limit jejího ověření.

Popis kohorty by měl doplnit zdroj a důvody CT vyšetření, kritéria výběru a vyloučení, nakládání s degenerativními změnami SIJ a vztah k předchozím souborům. Úplná data pro zahrnutých 278 subjektů sama nedokládají reprezentativnost ani absenci selekce. Kvadratická věková citlivost v supplementu je užitečná, ale neřeší nemodelované degenerativní či hormonální změny.

**9. Argumentační logika a kompozice.**

Článek má srozumitelnou návaznost anatomie → model → zatížení → kinematika → anatomické asociace. Výsledky, tabulky a závěr jsou z velké části konzistentní a diskuse obvykle respektuje rozsah měření. Silné je přiznání retrospektivního výběru hypotéz, částečně nepodpořené H2, neaditivity variance, absence ekvivalenčního testu a rozdílu podmíněného sklonu od uniformní podobnosti.

Slabinou je, že syntéza „scale–architecture–load path“ propojuje několik druhů evidence, aniž by je mechanisticky oddělila: objemové asociace jsou regresní; G zachovává zároveň size i shape; změna zatížení je skutečný kontrolovaný zásah do modelu. Výsledný příběh je přesvědčivější jako popis organizace modelované odezvy než jako vysvětlení jejího biologického mechanismu. Velký počet statisticky významných sklonů tento rozdíl neřeší; testy navíc sdílejí subjekty a korelované endpointy a nejsou dvaceti nezávislými replikacemi.

Úvod jmenuje širší biologické a porodnické souvislosti, ale výzkumná otázka zůstává široká. Hypotézy H3/H4 jsou poprvé plně vyloženy ve statistických metodách a škálování je exploratorní, přesto dominuje začátku diskuse. Doporučuji do konce úvodu dát jednu hlavní otázku, od ní odvodit předpovědi a diskusi vystavět podle důkazů potřebných pro její zodpovězení. Označení H1–H4 může zůstat, ale bez dojmu, že retrospektivní struktura představuje předem ověřovanou teorii.

**10. Vhodnost pro Journal of Theoretical Biology.**

Oficiální [popis Journal of Theoretical Biology od Elsevier](https://shop.elsevier.com/journals/journal-of-theoretical-biology/0022-5193) zahrnuje počítačové simulace, biophysical modeling i statistickou analýzu, pokud přinášejí významný biologický poznatek a podporují či vyvracejí teoretické myšlenky. Nová matematická metoda není nezbytnou podmínkou; rozhodující je biologický význam. Biomechanické FE studie proto do časopisu mohou patřit.

Relevantním precedentem je [Dumont, Grosse a Slater, 2009, *Requirements for comparing the performance of finite element models of biological structures*](https://doi.org/10.1016/j.jtbi.2008.08.017), který spojuje FE srovnání s oddělováním velikosti a tvaru a s definicí mechanického výkonu. Je vhodné jej do zdejší diskuse srovnávacího škálování zahrnout; jeho konkrétní pravidla pro stres a energii však nelze bez odvození převzít jako pravidla pro SIJ kinematiku.

Můj editorsky orientovaný odhad je **hraniční vhodnost současného konceptu**, nikoli předpověď rozhodnutí konkrétní redakce. Rukopis zatím především mapuje populační odpovědi a ukazuje citlivost na adjustaci a hustotní zjednodušení. Není jasné, jaký nově testovaný biologický princip vyplývá z pozorování, že větší a anatomicky odlišné pánve se při různém zatížení deformují jinak. Pro JTB by bylo silnější vysvětlit, které aspekty architektury a tkáňových poměrů určují load-specific compliance, nebo přesvědčivě otestovat hypotézu, že zdánlivé pohlavní kontrasty jsou v tomto modelu reprodukovatelné z velikosti a architektury napříč zatíženími.

Samotné přidání rovnic, změna názvu nebo silnější biologické formulace by tuto mezeru nevyřešily. Kontrolované geometrické perturbace, oddělení size/shape a citlivost materiálové hierarchie by naopak mohly zvýšit teoretický přínos. Pokud zůstane rozsah převážně populační funkční anatomie a kvantifikace modelového zjednodušení, podle mého hodnocení by po opravě numeriky byl přirozenějším kandidátem anatomický nebo biomechanický časopis. Toto je závěr z obsahu, nikoli ověření podmínek konkrétního alternativního časopisu.

**Doporučené pořadí revize.**

1. **Vyřešit geometrické inverze a doložit jejich dopad.** Bez toho nemá smysl finálně stavět mechanistický příběh na současných FE výsledcích.
2. **Doplnit nejbližší předchozí publikaci a původ kohorty.** Jasně napsat, co je společné a co je nové.
3. **Vymezit hlavní biologickou otázku.** Zvolit mezi podmíněným populačním popisem a skutečným testem velikosti/tvaru/kompliance; rozsah musí odpovídat závěru.
4. **Doložit citlivosti potřebné pro hlavní tvrzení.** Přípustné mapování a numerická přesnost mají přednost; potom hustotní zákon, kloubní tkáně a pretension, anatomická adjustace a překryv skupin. Konkrétní rozsah citlivostí má vycházet z konečné hlavní otázky.
5. **Doplnit přímá srovnání a distribuci chyb.** LAB1–LAB2 lze podpořit existujícími endpointy; RMSE doplnit alespoň o percentily a maximální chyby. Odlišné sklony mezi zatíženími testovat s využitím pairing, pokud mají být součástí závěru.
6. **Zveřejnit odvozené výstupy a reprodukční podklady** a teprve potom dokončit výběr cílového časopisu a jeho formát.

**Formulace závěru odpovídající nynější statistické evidenci**, po vyřešení numerické přípustnosti vstupů:

„Within the specified passive static model, SIJ kinematic responses showed inverse conditional associations with pelvic bone volume and depended on loading configuration. Retaining individual geometry with a population-median density field closely reproduced the full-model endpoints, while adjustment for volume and selected morphometric dimensions attenuated sex-associated translation contrasts under localized loading. These findings support anatomically resolved comparisons under controlled loading; they do not establish a universal scaling law or equivalence of female and male pelvic mechanics.“

Výstupy tohoto posudku: `mapping_summary.json`, `mapping_quality.csv`, `mapping_nonpositive_cells.csv`, `statistics_audit_summary.json`, `volume_model_audit.csv`, `agreement_audit.csv`, `exploratory_site_contrasts.csv` a reprodukční skript `audit_statistics.py` jsou v této složce. Citovaná čísla nových diagnostik jsou podmíněná stávajícími archivovanými FE řešeními. Rukopis, jeho obrázky a primární tabulky nebyly změněny.
