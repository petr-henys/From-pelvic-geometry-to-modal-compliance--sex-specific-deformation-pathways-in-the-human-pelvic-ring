# Interní odborná kontrola rukopisu BMMB (28. 9. 2026)

Kontrola zahrnula rukopis, uložené kohortové FE výsledky, simulační metadata, analytické skripty, bibliografii a nově provedené numerické přepočty. Tento dokument není suplementem článku a není v balíčku pro časopis.

## Opravené numerické body

- **Kostní materiálový zákon:** Implementovaný materiálový modul vzniká aplikací prahového mocninného zákona na zdrojové hustoty a teprve poté RBF interpolací na FE buňky. Původní citlivost používala opačné pořadí. `simulation/data_mapper.py` nyní interpoluje analytické derivace zdrojového modulu. Nezávislé konečné diference na 1669 buňkách subjektu 0 dávají relativní chybu $1,95\times10^{-11}$ pro α a $3,66\times10^{-11}$ pro β. Přepočet prvního řádu ze všech 278 uložených subjektových vlastních vektorů je dokončen pomocí `analysis/recompute_bone_sensitivities.py`. Přímá kontrola DOLFINx integrálů u módů 1, 9 a 15 subjektu 0 souhlasí s vektorovým výpočtem na relativní úrovni $10^{-13}$. Nová nejistota je promítnuta do hlavní tabulky a obrázku.
- **Kapacita bloků s více módy:** Původní jediné spuštění Nelder–Mead pro rozměr ≥3 nedávalo v některých případech optimum. `analysis/spectral_metrics.py` nyní řeší konvexní úlohu s uzlovými normovými omezeními, generuje aktivní omezení a vyhodnocuje celé pole uzlů. Původní kontrolní vzorek ukázal chybu až 0,00745 mm/mm u 12–15; kohortový přepočet aktualizoval všech pět anatomických kapacit bloku 12–15 a závislé grafy a tabulky. Nezávislá plně omezená optimalizace čtyř subjektů souhlasí do $1,9\times10^{-13}$ mm/mm.
- **Atlas:** Referenční atlas, jeho profily a tabulka byly vytvořeny znovu 14bodovou kvadraturou stupně 4 pomocí `analysis/reference_mode_atlas.py`. Uložená provenience nyní odpovídá použitému kódu. Stupeň 4 integruje všechny součiny kvadratického slovníku přesně po elementech.

## Dříve opravené obsahové body

- Mesh má 166 808 tetraedrů, 48 291 uzlů a 144 873 posuvových DOF. Izotropní velikostní rozdíly v kohortě nebyly v hlavní spektrální analýze odstraněny normalizací.
- BIS označuje šířku mezi trny sedacích kostí v midpelvis, BIT intertuberózní outlet. Anatomické funkcionály používají 32 referenčních uzlů a váhy vzdálenost$^{-2}$.
- Vlastní čísla mají při použité objemové metrice jednotky N/mm$^4$, jejich převrácené hodnoty nejsou bodové poddajnosti. SVD singularity nevyjadřují automaticky kapacitu při maximu uzlového posunutí 1 mm.
- Archivovaný inkrementální test předpětí má 18 validních subjektů. Nejvíc rekrutovaný mód souhlasí u všech pěti zatížení; velikost dominantní sady se u LAB1 liší ve 4/18.
- Bootstrap prevalence clusterů a intervaly srovnání ranků používají různé výběrové postupy, které jsou v textu nyní odděleny.
- Srovnání podle výměny módů jsou popisná, nikoli příčinná. Menší změna kapacity neprokazuje biologickou ekvivalenci. Rešerše přiznává přímé předchůdce pánevní spektrální analýzy i obecnou metodu shlukovaných podprostorů.

## Ověřené závěry a zbývající meze

Shoda hashů sedmi klíčových vstupů s uloženou proveniencí byla potvrzena. Kohorta má 278 osob (150 žen, 128 mužů). Vlastní čísla jsou kladná a seřazená. Prahový cluster 9–10 je přítomen u 152 osob. Původní nezávislý přepočet potvrzuje hlavní mediány 0,454→0,139 mm/mm pro rank 9 a 0,551→0,525 mm/mm pro AP kapacitu bloku 9–10; intervaly, p-hodnoty a BH q-hodnoty odpovídají zdrojovým datům. Všech 17 existujících matematických/statistických testů prošlo.

Energetické podíly jsou normalizovány jen vůči 15 zachovaným módům. Archivovaná kontrola širšího spektra na třech subjektech zachytila proti plné kvadratické energii 99,3–99,5 % pro stoj a 76,5–94,9 % pro expanzní síly; neprokazuje dostatečnost pro celou kohortu. Pevná sakrální penalizace 10$^6$ N/mm$^3$ nemá dohledatelný citlivostní sweep. IV allometrie používá instrumenty odvozené z geometrie vstupující přímo do FE odezvy, takže její koeficienty nejsou automaticky příčinné. Experimentální validace anatomických funkcionálů a modelování porodnických stavů zůstávají mimo rozsah této studie.
