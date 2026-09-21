# Kontrola referencí — 20. 9. 2026

Zkontrolováno všech 15 původních záznamů; doplněny čtyři věcně potřebné zdroje. Konečný stav: **19 citací, 19 unikátních DOI registrovaných v Crossref, žádný chybějící ani nepoužitý citační klíč**. Platnost znamená shodu registrovaného DOI s publikací; nezaručuje volný přístup k plnému textu ani dostupnost webu vydavatele při každém požadavku.

## Podstatné opravy

- Keller: Thomas S. → Tony S.; Masi: neověřené rozvedení „Anthony“ nahrazeno doloženými iniciálami A. T.
- BoneDat: doplněno číslo článku 1043; Heyland: číslo časopisu 11.
- Doplněna DOI bootstrapové knihy a Benjaminiho–Hochbergovy práce. Bonet–Wood převeden z prvního vydání 1997 na druhé vydání 2008 s odpovídajícím DOI; DOI druhého vydání nebylo připojeno k nesprávnému roku.
- Kellerův článek již nepůsobí jako zdroj celé implementované materiálové aproximace. Jeho abstrakt navíc jasně uvádí závislost fitu na rozsahu hustot; neopravňuje automaticky přenést libovolný fit na pánev.
- Vložené čtyři zdroje podporují konkrétní anatomické či metodické výroky, nikoli obecné zdání rozsáhlejší rešerše.
- Wilcoxonův dosud nepoužitý záznam nyní odkazuje pouze na skutečně reportovanou pomocnou diagnostiku.
- Zachovány správné roky časopiseckých vydání Pavličev 2020 a Henyš–Hammer 2025, přestože online zveřejnění bylo dřívější. Neobvyklé „in silicon modelling“ u Heylanda je skutečný publikovaný název a nebylo svévolně opraveno.
- Lokální odvozený styl `bib/apalike-doi.bst` zachovává formát apalike a tiskne klikatelné DOI.

## Kontrola použití po jednotlivých položkách

U bibliografických metadat byly porovnány identita publikace, autoři, název, rok, časopis/kniha, svazek a dostupné stránkování. Hloubka obsahové kontroly se liší podle dostupnosti textu; tabulka ji výslovně uvádí. Kontrola abstraktu není vydávána za přečtení celé práce.

| Klíč | Ověřený DOI | Použití a závěr | Rozsah obsahového ověření |
|---|---|---|---|
| `Goode2008` | [10.1179/106698108790818639](https://doi.org/10.1179/106698108790818639) | Přehled malých pohybů SIJ a obtíží jejich měření; nikoli validace našich amplitud. | Plný text PMC2565072. |
| `Vleeming2012` | [10.1111/j.1469-7580.2012.01564.x](https://doi.org/10.1111/j.1469-7580.2012.01564.x) | Anatomie, svalová komprese, vazy a přenos zatížení; odpovídá úvodu a diskusi. | PubMed/Europe PMC abstrakt; metadata. |
| `Snijders1993` | [10.1016/0268-0033(93)90002-Y](https://doi.org/10.1016/0268-0033(93)90002-Y) | Teorie samosvornosti a stability SIJ; použita jako mechanický kontext, ne důkaz našich výsledků. | Primární abstrakt; metadata. |
| `Keller1994` | [10.1016/0021-9290(94)90056-6](https://doi.org/10.1016/0021-9290(94)90056-6) | Experimentální motivace vztahu hustota–tuhost. Citace přesunuta před implementovaný zákon; převod hustoty, exponent a dolní limit nejsou vydávány za doslovný Kellerův zákon. | Primární abstrakt (496 vzorků z pěti mužských dárců); metadata. Celý článek nebyl zkontrolován. |
| `ChenGrimm2021` | [10.1115/1.4049226](https://doi.org/10.1115/1.4049226) | Rozsah modelů porodu a potřebná anatomie/tkáně; ne důkaz porodní relevance našich zatěžovacích případů. | Primární abstrakt; metadata. |
| `HenyssKuchar2025BoneDat` | [10.1038/s41597-025-05161-y](https://doi.org/10.1038/s41597-025-05161-y) | Zdroj standardizované morfologie a 278 CT; nikoli nezávislá validace kinetiky. Doplněno číslo článku 1043. | Stránka a text vydavatele Nature; abstrakt; metadata. |
| `BonetWood2008` | [10.1017/CBO9780511755446](https://doi.org/10.1017/CBO9780511755446) | Transformace gradientů a integračních měr; obecný matematický zdroj, nikoli validace našeho mapování. Vědomě změněno na druhé vydání 2008, ke kterému patří DOI. | Záznam vydavatele a Crossref, popis knihy; nikoli celý text knihy. |
| `Wilcoxon1945` | [10.2307/3001968](https://doi.org/10.2307/3001968) | Zdroj signed-rank metody, doplněný přímo u pomocné diagnostiky; nepřipisuje se mu primární binomický znaménkový test. | Bibliografická kontrola původní metodické práce. |
| `EfronTibshirani1994` | [10.1201/9780429246593](https://doi.org/10.1201/9780429246593) | Bootstrapové intervaly a párové převzorkování; ne zahrnutí modelové nejistoty. DOI odpovídá vydání 1994, nikoli alternativnímu Springer záznamu 1993. | Záznam knihy vydavatele a Crossref; nikoli celý text knihy. |
| `MacKinnonWhite1985` | [10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7) | Historický metodický zdroj heteroskedasticitně robustních odhadů HC3; nevztahuje se ke clusterování opakovaných měření. Moderní HC3 je obvyklá zjednodušená varianta jackknife odhadu. | Metadata a abstrakt; terminologie ověřena také v autorském přehledu MacKinnona, viz odkaz níže. |
| `BenjaminiHochberg1995` | [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) | Metoda úpravy p-hodnot v popsaných rodinách testů. Nejde o záruku kontroly FDR při libovolné závislosti testů. | Metadata původní metodické práce; DOI doplněno. |
| `Hammer2013` | [10.1016/j.spinee.2013.03.050](https://doi.org/10.1016/j.spinee.2013.03.050) | Vliv pánevních vazů a chrupavky na stabilitu; nelze jím dokládat zanedbatelnost materiálových parametrů. | Primární strukturovaný abstrakt; metadata. |
| `Pavlicev2020` | [10.1016/j.ajog.2019.06.043](https://doi.org/10.1016/j.ajog.2019.06.043) | Evoluční/porodnický kontext, nikoli potvrzení jednoduchého kompromisu šířka pánve–energetická náročnost chůze. Rok 2020 je správně podle čísla časopisu. | Primární abstrakt; metadata. |
| `HenysHammer2025` | [10.1111/joa.14160](https://doi.org/10.1111/joa.14160) | Předchozí vztah morfologie SIJ a mechanické odezvy; explicitně označen jako předchozí práce. Soubor 281 modelů se nesmí prezentovat jako nezávislá validace současných 278. | Primární abstrakt a autorský institucionální záznam; metadata. |
| `Heyland2025` | [10.1007/s11517-025-03396-w](https://doi.org/10.1007/s11517-025-03396-w) | Citace podporuje význam předpětí vazů. Studie používá typický ženský a mužský FE model odvozený z většího CT souboru, nikoli 818 FE modelů. Doplněno číslo 11. | Primární abstrakt; stránka vydavatele; metadata. |
| `Huseynov2016` | [10.1073/pnas.1517085113](https://doi.org/10.1073/pnas.1517085113) | NOVĚ: vývojové a pohlavní rozdíly rozměrů pánve v úvodu; nikoli důkaz pohlavních rozdílů pohyblivosti SIJ. | Plný text PMC4868434; metadata. |
| `Anderson2005` | [10.1115/1.1894148](https://doi.org/10.1115/1.1894148) | NOVĚ: konkrétní příklad experimentální validace pánevního modelu pomocí kortikálních deformací. Výslovně odlišeno od validace našich SIJ kinematických výstupů. | Primární abstrakt PubMed16060343; metadata. |
| `Kabsch1976` | [10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873) | NOVĚ: základní rigidní least-squares registrace. Trimming a výběr ROI výslovně označeny jako implementační volby. | Původní vydavatelský záznam IUCr/Wiley a metadata; celý text nedostupný. |
| `CameronMiller2015` | [10.3368/jhr.50.2.317](https://doi.org/10.3368/jhr.50.2.317) | NOVĚ: cluster-robustní kovariance při závislosti měření uvnitř subjektu. Správně přiřazeno ke sdruženým modelům s 278 clustery. | Autorský plný text, abstrakt a metodický úvod; metadata. |

## Doplňující doklady a meze

- [MacKinnonův autorský přehled robustní kovariance](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/qed_wp_1268.pdf) rozlišuje původní jackknife a dnes běžnou HC3 variantu. Původní metodická citace je přiměřená, není však důkazem konkrétní softwarové implementace.
- [Cameron–Miller, autorský rukopis](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf) dokládá potřebu připustit korelaci uvnitř clusteru.
- Přesný původ nominálních parametrů chrupavky, vazů, předpětí a převodu hustoty není doložen externí experimentální kalibrací. Text je správně označuje jako implementované/archivované volby; přidání nesouvisející citace by tento problém nevyřešilo.
- Dokumentace překryvu kohorty s předchozí studií zůstává v metodách otevřeným požadavkem. Bibliografická kontrola nenahrazuje doložení této provenance.
- BH úprava má předpoklady o závislosti testů. Audit potvrzuje správný metodický zdroj a deklarované rodiny, nikoli obecnou záruku FDR pro libovolnou korelaci výstupů.

## Reprodukovatelnost a kontrola sestavení

Surová metadata jsou v `references/crossref_verified.json` a `references/crossref_additions.json`; primární abstrakty v `references/europepmc_records.json`. Starší soubor `crossref_original.json` zachovává první pokus včetně omezení rychlosti 429; rozhodující jsou úspěšné odpovědi uložené v ověřených záznamech.

Po úpravách: latexmk/BibTeX úspěšně vytvořil `main.pdf`; bez nedefinovaných citací a bez Overfull/Underfull hlášení. Zkontrolována úplnost všech 19 DOI v bibliografickém výstupu. Číselné výsledky ani obrázky nebyly touto revizí změněny.
