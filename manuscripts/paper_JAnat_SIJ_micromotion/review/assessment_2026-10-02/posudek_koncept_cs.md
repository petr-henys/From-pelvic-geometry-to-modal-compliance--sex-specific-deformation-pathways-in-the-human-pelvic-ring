**Věcný a konceptuální posudek SIJ_micromotion — 2. října 2026**

Posouzení se zaměřuje na originalitu, argumentaci, statistickou interpretaci, vztah výsledků k závěrům a vhodnost pro Journal of Theoretical Biology.

**Verdikt:** Článek má samostatný publikovatelný obsah a jeho hlavní omezené závěry jsou převážně podpořeny. Originalita je nejsilnější v kombinaci populační kinematiky, více zatížení a párovaného hustotního zjednodušení. Současná verze ale spíše popisuje anatomicky podmíněné modelované odpovědi, než vysvětluje jejich biologický mechanismus. Pro JTB bych doporučil zásadní konceptuální revizi; samotná jazyková úprava nebo rozšíření matematického appendixu tento rozdíl nevyřeší.

**1. Co je skutečně nové a co už bylo známo.**

V rukopisu chybí nejbližší předchozí práce stejných autorů: [Henyš a Hammer (2025), *Sacroiliac joint auricular surface morphology modulates its mechanical environment*](https://doi.org/10.1111/joa.14160). Ta již použila 281 populačních FE modelů a analyzovala vztah morfologie SIJ k jeho mechanickému prostředí. Je nutné uvést vztah kohort a přesně vymezit nové endpointy, zatížení a otázku. Shodné využití kohorty není samo problém; nesmí ale vytvářet dojem nového nezávislého populačního důkazu. Konkrétní překryv subjektů nebyl při tomto posouzení ověřen.

Nový přínos může spočívat v odhadu relativního pohybu SIJ, jeho load-specific variabilitě a přesnosti varianty se zachovanou geometrií a standardizovanou hustotou. Podstatná je i otázka, kolik informace o pohlavním kontrastu obsahuje malá sada anatomických proměnných. Samotné pozorování větší asymetrie při jednostranném zatížení nebo menšího pohybu u větších struktur má slabší konceptuální originalitu. Rozsah 278 geometrií zvyšuje sílu populačního popisu, ale sám nezakládá nový biologický princip.

**2. Pohlaví není nezávislým vstupem FE modelu.**

Tato skutečnost by měla být při interpretaci H3 vysvětlena přímo. Model má individuální geometrii a hustotu, ale pohlavně specifické vlastnosti vazů, chrupavky, pretension nebo svalové stabilizace nejsou zadány. Pohlaví je až statistická proměnná označující skupiny subjektů.

Všechny simulované pohlavní rozdíly tedy vznikají z rozdílů skutečných vstupů modelu. Zbytkový regresní koeficient pohlaví může zachytit nepozorované geometrické či hustotní charakteristiky nebo nedokonalou statistickou specifikaci; není odhadem nějakého přímého biologického účinku pohlaví na tkáně. Vymizení jeho významnosti po přidání anatomie zase nedokazuje, že v reálné populaci neexistují další pohlavní rozdíly v mechanice.

Proto není nejsilnější otázkou „zda anatomie vysvětluje pohlaví“. V této implementaci anatomie a hustota tvoří mechanismus rozdílů už konstrukcí modelu. Podstatnější otázka zní: **Dokáže malá sada interpretable anatomických měr zachytit rozdíly plného modelu a určit, jak se projeví při různých zatíženích?** Pro její zodpovězení by pomohlo uvést přírůstek vysvětlené variability, predikční přínos triády a stabilitu vztahů na zadržených subjektech.

Nynější tvrzení, že anatomická adjustace attenuuje pohlavně asociované LAB translace, je přesto věcně správné. Koeficienty M0→M2 se mění +0,055→+0,013; +0,071→−0,031 a +0,258→+0,002 mm. Jde o výsledek konkrétního podmíněného srovnání, který zasluhuje opatrnou formulaci použitou v současném závěru.

**3. Pohlavní kontrasty závisejí i na specifikaci statistického modelu.**

Samostatné LAB modely mají všech šest plně adjustovaných intervalů pohlavních koeficientů přes nulu. Pooled modely v Table S5 však uvádějí ženský koeficient translace +0,046 mm s q = 0,045 a interakci pohlaví×objem pro rotaci s q = 0,033. Naproti tomu load-specific log–log interakce mají po korekci q > 0,34.

To není formální nekonzistence: modely používají jiné škály odpovědi a jiné předpoklady sdílení anatomických efektů napříč zatíženími. Pooled pohlavní koeficient s interakcí je navíc podmíněný hodnotou objemu. Článek by ale měl čtenáři stručně vysvětlit, proč tyto výsledky nejsou zaměnitelné. Nelze současně naznačovat všeobecnou nepodstatnost pohlaví nebo univerzálně stejné sklony.

Výrazné rozdělení anatomických proměnných mezi pohlavími také znamená, že „adjustovaný rozdíl“ nemusí být reprezentativním srovnáním typické ženy s typickým mužem. Doporučuji doložit společnou podporu kovariát a citlivost na zdůvodněné alternativy triády. VIF 5,55 tuto otázku sám neřeší. Retrospektivní výběr tří z osmi měr musí být popsán tak, aby bylo jasné, zda vycházel z anatomické úvahy, dostupných korelací nebo předchozího prozkoumání výsledků.

**4. Škálování: asociace je podpořena, mechanický zákon prokázán není.**

Dimenzionální vztahy u ∝ F/(EL) a θ ∝ F/(EL²) dávají při V ∝ L³ referenční sklony −1/3 a −2/3. Jde však o uniformní změnu geometricky podobné struktury. Nynější regrese drží vybrané vzdálenosti v milimetrech konstantní, zatímco mění objem. Při takovém podmínění se kromě velikosti mění také vztah objemu k fixovaným rozměrům; koeficient nemá význam čisté homotetické změny.

Aktuální text i Figure 7 tento rozdíl výslovně uvádějí. Nedoporučuji jej proto označovat za chybnou matematiku. Slabší je argumentační váha přikládaná numerické blízkosti některých sklonů k referencím, protože srovnávané hodnoty odpovídají jiným estimandům.

Diagnostika posudku držela věk, pohlaví i interakci objem×pohlaví a měnila pouze přítomnost triády. U LAB1 translace žen se sklon změnil z −0,481 bez triády na −1,003 s triádou; u LAB2 translace mužů z −0,382 na −0,818. Všech dvacet bodových odhadů zůstalo záporných i bez triády, ale ne všechny zůstaly významné po stejné BH korekci. To ukazuje stabilitu směru a zároveň silnou závislost velikosti sklonů na podmínění.

Pro JTB je zajímavé vysvětlit právě tuto závislost. Strmější LAB sklony mohou souviset se změnou proporcí, momentových ramen, tuhosti kloubních tkání nebo samotnou adjustací. Současná diskuse tyto možnosti vypočítává, ale nerozlišuje mezi nimi. Buď je potřeba jednu mechanismovou hypotézu otestovat, nebo ponechat škálování jako exploratorní statistický výsledek a nedělat z referenčních exponentů hlavní teoretickou osu.

**5. Název a syntéza rozlišují size a architecture důsledněji než experimentální design.**

Architektura je v úvodu definována jako individuální geometrie a proporce nad rámec velikosti. Varianta G ale zachovává obojí současně. Její shoda s full proto dokládá význam celé individuální geometrie, nikoli samostatný vliv tvaru při stejné velikosti.

Koeficient objemu v adjustované regresi a shoda G/full jsou různé druhy evidence. Jejich kombinace je užitečná, ale nevytváří experimentální separaci size/shape. Pokud má „architecture“ znamenat samostatný mechanismus, vhodná by byla velikostně normalizovaná geometrická varianta nebo cílené změny konkrétního anatomického znaku. Pokud takové analýzy nejsou součástí práce, doporučuji v názvu a diskusi klást důraz na **individual geometry** a na anatomické asociace.

Dobrý teoretický model může být jednoduchý. Potřebuje ale vysvětlit, proč například změna proporcí nebo lokální geometrie SIJ ovlivňuje konkrétní komponentu pod jedním zatížením a méně pod jiným. Vztah „různá geometrie + různé síly → různé pohyby“ je obecně očekávatelný. Novost musí být v určení rozhodujících anatomických vztahů nebo v nové testovatelné predikci.

**6. Hustotní zjednodušení je nejsilnější praktický výsledek.**

Minimum identity-line R² translace 0,9457 a maximum RMSE 0,0280 mm jsou přesvědčivé ukazatele shody G/full v dané implementaci. Je správné, že analýza nepracuje pouze s poměry variance a že mediánová hustota zůstává prostorově proměnná.

Interpretace však musí rozlišit tři otázky: variabilitu použitého hustotního pole, citlivost kinematiky na z něj odvozený modul a platnost zjednodušení při alternativních tkáňových parametrech. Nízkohustotní větev E = 2398 MPa potlačuje část variability modulu a nominálně stejné měkké tkáně omezují rozsah srovnání. Výsledek proto nelze rozšířit na obecnou nepodstatnost hustoty.

Pro použití individuálních predikcí je důležitý rozptyl chyb. Přepočtená maximální chyba translace při SP2leg je 0,2195 mm, přestože RMSE je 0,0280 mm; 95. percentil je 0,0486 mm. Shoda je tedy dobrá v souhrnu, nikoli uniformně do uvedeného RMSE. Závěr o zjednodušení by měl obsahovat toleranci podle zamýšleného využití a percentily individuálních chyb. Studie zatím nepotvrzuje klinickou přijatelnost.

Pro mechanistické zobecnění by nejvíce pomohla cílená citlivost na hustotně-modulový zákon a na relevantní rozsah chrupavkové, ligamentové a pretension variability. Není nutné automaticky rozšiřovat všechny analýzy; rozsah má odpovídat finálnímu hlavnímu tvrzení.

**7. Load path má dobrou podporu, jeho biologický výklad může být konkrétnější.**

Párované SP1leg/SP2leg porovnání je přesvědčivé: stejné 800 N celkového acetabulárního zatížení při jiném rozdělení zvětšuje pohyb a bilaterální asymetrii. Článek správně upozorňuje, že momenty zůstávají odlišné.

H2 je pouze částečně podpořena a rukopis to poctivě uvádí. Přímá exploratorní diagnostika LAB2−LAB1, doplněná v posudku z existujících endpointů, ukazuje mediány rozdílů absolutních komponent ML +0,0576 mm, AP +0,1108 mm a CC −0,0204 mm; všechny tři sign testy přežijí BH korekci ve společné třítestové rodině. Tuto podporu rozdílu míst aplikace lze doplnit do článku místo spoléhání na oddělená porovnání se SP2leg.

Biologický význam by posílilo vysvětlení momentových ramen a toho, proč se mění právě tyto komponenty. Absolutní velikosti však stále neidentifikují směr nutace, gapping ani změnu porodního kanálu. Aktuální text tento rozsah měření respektuje a měl by jej zachovat.

**8. Celková konzistence a vztah výsledků k závěru.**

Kontrolované regrese a souhrny jsou reprodukovatelné. Závěr, omezený na standardizované zatížení, podmíněné objemové asociace, hustotní zjednodušení a attenuaci LAB kontrastů, odpovídá datům. Nepodpořená by byla tvrzení o univerzálním škálovacím zákonu, obecné ekvivalenci pohlaví, kauzálním účinku triády nebo porodní funkci.

Příběh má čtyři podstatné výsledky, ale slabší hierarchii: H3/H4 jsou plně vysvětleny až v metodách, škálování je exploratorní a přesto dominuje začátku diskuse, zatímco nejpraktičtější výsledek G/full je prezentován hlavně jako další potvrzení široké anatomické teze. Doporučuji dát do konce úvodu jednu hlavní otázku a všechny výsledky vztáhnout k ní. Diskuse by měla rozlišovat doloženou odpověď, její možné vysvětlení a predikci pro další ověření.

**9. Konkrétní cesta k Journal of Theoretical Biology.**

Oficiální [zaměření JTB](https://shop.elsevier.com/journals/journal-of-theoretical-biology/0022-5193) připouští simulace a statistické analýzy při významném biologickém poznatku a podpoře či vyvracení teoretických myšlenek. Nevyžaduje novou matematickou metodu. Relevantním precedentem srovnávací biomechaniky je [Dumont et al. (2009)](https://doi.org/10.1016/j.jtbi.2008.08.017), který propojuje FE performance s oddělováním velikosti a tvaru. Jeho pravidla pro stres a energii však nelze bez odvození převzít pro zdejší kinematiku.

Nejslibnější otázku vidím v tomto směru: **Do jaké míry lze populační SIJ odezvu a její pohlavní kontrasty předpovědět z velikosti a omezeného počtu anatomických znaků napříč zatíženími, a jaký mechanický vztah určuje selhání této redukce?** Současné výsledky jsou dobrým začátkem, ale ještě neodpovídají na její predikční a mechanismovou část.

Pro posílení JTB verze bych prioritně doplnil:

1. Vymezení nové otázky a rozdílu proti dřívější populační práci.
2. Vysvětlení, že pohlaví v FE modelu označuje anatomické a hustotní vstupy, nikoli samostatné pohlavně specifické tkáňové vlastnosti.
3. Analýzu, co přidává triáda nad samotnou velikost: velikost přírůstku fitu, stabilitu predikce na zadržených subjektech a load-specific anatomické koeficienty.
4. Jeden cílený test mechanismu, například kontrolovanou změnu velikosti/tvaru nebo citlivost postulované geometricko-materiálové hierarchie.
5. Přímé porovnání relevantních load cases a rozdělení individuálních chyb G/full.

Můj odhad je, že tato cesta by mohla založit přesvědčivou JTB publikaci. Pokud článek zůstane u nynějšího deskriptivního rozsahu, bude jeho obsah přirozeněji čitelný jako populační funkční anatomie a biomechanika. Nejde o předpověď rozhodnutí redakce, ale o posouzení síly biologické otázky a důkazů.
