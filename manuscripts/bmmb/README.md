# Rukopis pro Biomechanics and Modeling in Mechanobiology

Samostatná redakční verze z `../natcomm/main_plos.tex`, připravená 25. 9. 2026, matematické metody rozšířeny 26. 9. 2026. Původní rukopis a analytické výstupy nejsou změněny. Obrázky, tabulky a číselná makra jsou lokální kopie, takže dokumenty lze sestavit mimo původní projekt.

## Dokumenty

- `main_bmmb.tex` / `main_bmmb.pdf`: jediný úplný rukopis; hlavní text a pět appendixů tvoří jeden článek. V PDF zůstává všech 15 obrázků.
- `cover_letter.tex` / `cover_letter.pdf` / `cover_letter.md`: průvodní dopis.
- `bmmb_submission.zip`: samostatný balíček rukopisu, dopisu a jejich zdrojových závislostí. Neobsahuje samostatný supplement.
- `asset_provenance.json`: původ převzatých podkladů a kontrolní součet původního rukopisu.

Veškerý zachovaný obsah dřívějšího supplementu je součástí jediného rukopisu. Appendixy A–E obsahují referenční polynomiální atlas, asociace anatomické kapacity a zátěžového směrování, alometrickou analýzu, citlivost na materiálové parametry a analytický model. Duplicitní diagnostiky a tabulky byly zkráceny; číslování obrázků a tabulek je průběžné.

## Hlavní redakční změny

Název: **Comparing deformation subspaces across anatomical variability: a population finite-element study of the human pelvis**.

Text staví na problému srovnatelnosti módů mezi jedinci, anatomických funkcionálech a statickém rozkladu energie. Rozlišuje kinematickou kapacitu od poddajnosti na jednotku síly a matematickou invarianci báze od empirické stability mezi jedinci. Obecná tvrzení o zachování biologické funkce a porodnické závěry jsou omezeny na to, co podporuje linearizovaný model.

Abstrakt má přibližně 215 slov, šest klíčových slov; sekce Author summary byla odstraněna. Použita je struktura Introduction–Materials and methods–Results–Discussion–Conclusions, samostatné Statements and Declarations. Hlavní Methods obsahují pullback FE formulaci, materiály a vazy, spektrální mezery, principal angles a Procrustes, anatomické funkcionály a optimalizaci kapacity i modální energie. Kvadratura, polynomiální slovník, Shapleyho rozklad, propagace nejistoty a rozšířený analytický model jsou v appendixech téhož článku. Atlas, rozšířené výsledky a validační kontroly zůstávají součástí jednoho PDF. Dva nové obrázky v Appendixu A ukazují 14bodovou kvadraturu na referenční síti a reprezentativní pole čtyř polynomiálních rodin deformací; jejich zdroj je `analysis/render_method_illustrations.py`. Po auditu byly přepočteny kapacity podprostorů, atlas a analytická propagace materiálové nejistoty z uložených vlastních vektorů; nové vlastní úlohy FE se neřešily. Schéma modelu a graf anatomických kapacit byly znovu vykresleny se sjednocenými popisky; nová mapa kapacity a zátěže používá stejnou definici kapacity jako hlavní výsledky.

Opravy podle existujících dat a kódu:

- Blok 5–6 není označen za blok s nejvyšší outlet kapacitou; zveřejněné hodnoty pro 9–10 jsou vyšší.
- Výsledek srovnání subspace capacity není vydáván za statistickou ekvivalenci: pro členy clusteru je p = 0.030 a q = 0.040.
- U hlavních intervalů byla opravena metoda na **5 000 percentilových bootstrapů** podle `analysis/plos_revision.py`, místo původního tvrzení 10 000 BCa.
- Kontrola oddělení předpětí je popsána jako shoda top módu ve 100 % proxy případů a 90 % případů stoje; změny podílů jsou v procentních bodech.
- Součet prvních tří podílů pro jednostranný stoj je sjednocen na 92.0 % podle existující tabulky.
- Analytická ilustrace přiznává podmínění výběru na zachovaný nízký mód; specifikace v hlavních Methods vychází přímo z `analysis/toy_subspace_dashboard.py`.
- `NaN` ve sloupci vnější separace je nahrazeno pomlčkou s vysvětlením truncation. Tabulky srovnání rank/subspace byly vysázeny z existujících CSV bez změny hodnot a bez zaokrouhlování malých p-hodnot na nulu.

## Co zbývá před skutečným odesláním

1. **Veřejný archiv kódu a odvozených dat zatím neexistuje.** Prohlášení to uvádějí pravdivě. Před odesláním dořešit přístup pro recenzenty a konečné znění Data/Code availability; pokud vznikne archiv, vložit DOI/URL i do průvodního dopisu. Současný balíček obsahuje podklady článku, nikoli kompletní reprodukční výpočetní prostředí.
2. Autoři mají zkontrolovat novou redakční verzi, afiliace, příspěvky, financování, střety zájmů a etické údaje převzaté z původního rukopisu. Případné členství autora v redakční radě se uvádí v systému časopisu.
3. Po potvrzení autorů doplnit do cover letter standardní prohlášení o původnosti, nepřítomnosti souběžného posuzování a souhlasu všech autorů. Nový dopis tyto dosud nepotvrzené skutečnosti nepředstírá.
4. Ověřit financování APC: oznámení časopisu zahrnuje podání od 11. 8. 2026 při přijetí, s případnými výjimkami. ORCID lze doplnit, pokud jsou k dispozici.
5. Bibliografické údaje byly převzaty z původní knihovny; nově byly ověřeny a doplněny práce o populačním modelování, pánevních FE kohortách, modální analýze a mechanických spektrálních metrikách (včetně přímého předchůdce Henyš et al., 2021). Toto zpracování nepředstavuje nezávislý audit všech bibliografických záznamů ani nové experimentální ověření modelu.

## Sestavení

V této složce spusťte:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main_bmmb.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter.tex
python3 package_submission.py
```

Balíček obsahuje lokální závislosti, oficiální třídu a bibliografický styl. Při uploadu označit `main_bmmb` jako hlavní rukopis a dopis jako Cover letter. README ani soubory provenance nejsou součástí vědeckého textu.

## Ověřené pokyny

- [Instructions for Authors](https://link.springer.com/journal/10237/submission-guidelines)
- [Aims and scope a oznámení přechodu na OA](https://link.springer.com/journal/10237/aims-and-scope)
- [Oficiální šablona Springer Nature](https://www.springernature.com/gp/authors/campaigns/latex-author-support), vydání December 2024, staženo 25. 9. 2026. `sn-jnl.cls` a `sn-basic.bst` jsou převzaty beze změn.

## Rozšíření rešerše 26. 9. 2026

Úvod byl přepracován s explicitním rozlišením statistických tvarových módů, mechanických módů, anatomické kapacity a zatížením vyvolané odezvy. Doplněno sedm bibliografických záznamů ověřených u vydavatelů nebo v PubMed: Cook a Robertson (2016), Salo et al. (2017), Arand et al. (2019), Henyš a Čapek (2019), Henyš et al. (2021, 2022), Chen et al. (2026). Existující Ghosh a Ghanem (2012) výslovně vymezuje matematický precedens. Diskuse a dopis návaznost reflektují; rešerše není označena za systematickou ani nepodkládá tvrzení o absolutním prvenství.

## Odborný audit 28. 9. 2026

Viz [audit/REPORT.md](audit/REPORT.md). Opraveny jsou derivace kostního zákona, výpočet čtyřmódových kapacit a numerická provenience atlasu. Interní auditní soubory nejsou suplementem ani součástí balíčku pro časopis.

## Redakční přesun 29. 9. 2026

Pět vedlejších analytických větví bylo přesunuto do appendixů téhož rukopisu. Hlavní text se soustředí na spektrální clustery, anatomickou kapacitu a zátěžové směrování. Redukovány byly druhé bootstrapové schéma, doplňkové tabulky výměn a morfologického permutačního testu, překryv profilů zátěží a duplicitní celokohortová tabulka; klíčové číselné výsledky zůstávají v textu. Appendixy nejsou samostatné soubory ani supplement.

## Kontrola matematického zápisu 29. 9. 2026

Sjednoceno značení referenčních a fyzických souřadnic landmarků, vážených kvadraturních vektorů, rodin polynomů, SVD indexů, modálních energií a bezrozměrných logaritmických citlivostí. Opraveny lokální proměnné analytického modelu a zpřesněno, že modalní energie zahrnuje všechny členy linearizované tuhosti. Číselné výsledky nebyly přepočítány.
