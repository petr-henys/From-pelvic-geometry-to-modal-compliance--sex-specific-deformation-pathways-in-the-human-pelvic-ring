# Formální kontrola pro Journal of Theoretical Biology

Kontrola provedena 2. října 2026. Rukopis i přílohy byly upraveny a znovu
exportovány do PDF a Wordu. Ověřené obecné požadavky Elsevier a níže uvedené
technické kontroly jsou splněné. **Úplnou shodu s aktuálními pravidly JTB nelze
potvrdit**, protože současný Guide for Authors nebyl dostupný. Zvlášť zůstává
otevřený aktuální limit délky a časopisecký požadavek na zpřístupnění odvozených
dat a kódu. Žádné soubory nebyly odeslány ani zveřejněny.

## Provedené úpravy a kontroly

| Položka | Stav |
| --- | --- |
| AI při přípravě rukopisu | Doplněno prohlášení bezprostředně před literaturu. Uvádí OpenAI Codex (GPT-6, OpenAI), rešerše, organizaci a psaní textu i vývoj analytického a vizualizačního kódu. Potvrzuje kontrolu a odpovědnost autorů. |
| AI ve výzkumném postupu | Doplněna podkapitola Methods: konkrétní pomoc při postprocessingu, výpočty z archivovaných dat, reprodukovatelné numerické grafy a nezávislé ověření výsledků. |
| Další AI nástroje | Autor potvrdil, že použity nebyly. |
| Střet zájmů | Prohlášení v rukopisu a samostatný editovatelný soubor `submission/Declaration_of_competing_interest.docx`. Zachováno původní tvrzení o nepřítomnosti střetu zájmů. |
| Financování | Poskytovatel a číslo grantu uvedeny; doplněno autorem potvrzené tvrzení, že poskytovatel neměl roli v návrhu, analýze, interpretaci, psaní ani rozhodnutí o publikaci. |
| Autorské příspěvky | Přítomné role CRediT obou autorů. |
| Schválení autory | Autor potvrdil kontrolu a schválení textu i výstupů oběma autory; potvrzení uloženo v `author_confirmations.json`. |
| Etika | Methods obsahují instituci, číslo schválení 202411IO3P a waiver of informed consent. Původní údaje zachovány; kontrolována jejich přítomnost. |
| Titulní údaje | Oba autoři, afiliace, poštovní adresy a e-mail korespondenčního autora přítomné. |
| Highlights | Samostatný Word, čtyři body; délky 76, 73, 76 a 74 znaků včetně mezer. Splňují ověřené pravidlo Elsevier 3–5 bodů, nejvýše 85 znaků na bod. |
| Abstrakt | Rozvinuty zkratky computed tomography, anteroposterior, RMSE a confidence interval; 239 slov. |
| Klíčová slova | Pět výrazů; neopakují doslova slova názvu. |
| Obrázky | Osm hlavních obrázků, popisky a odkazy; samostatná popisková stránka ve Wordu a PDF kopie `submission/figures/Figure_01.pdf` až `Figure_08.pdf` ve správném pořadí. Mapování v `submission/figure_files.csv`. |
| Tabulky | Pět hlavních tabulek; ve Wordu skutečné editovatelné tabulky. Příloha obsahuje 13 číslovaných tabulek, rozdělených do 15 nativních tabulek Wordu, a tři obrázky. |
| Literatura | 28 položek; množina citovaných klíčů odpovídá bibliografii. Žádné nevyřešené odkazy při exportu. |
| Formát PDF | Čísla řádků a stran, dvojité řádkování hlavního textu. Oba LaTeX buildy bez varování a přetékajících řádků. |
| Fonty a obrazová data | Devět starších PDF obrázků mělo Type 3 fonty. Text převeden do vektorových obrysů; nové exporty používají vložené TrueType fonty. Kontrola zahrnula oba výsledné PDF a všech 11 použitých obrázků. Dekódované bitmapy zůstaly přesně shodné s původními; vykreslení porovnáno. Záznam v `figure_font_normalization.json`. |
| Numerické výsledky | Patnáct numerických artefaktů evidovaných v předchozím manifestu má nezměněné SHA-256. Formální úpravy nevyžadovaly přepočet analýz. |
| Prázdné sekce | Odstraněna prázdná sekce Acknowledgements. |

## Počet slov a rozsah

| Část | Počet |
| --- | ---: |
| Název | 9 slov |
| Abstrakt | 239 slov |
| Introduction | 629 slov |
| Methods | 2 138 slov |
| Results | 1 125 slov |
| Discussion | 1 308 slov |
| Conclusion | 103 slov |
| Hlavní text bez abstraktu a Appendix A | **5 303 slov** |
| Hlavní text včetně abstraktu | 5 542 slov |
| Appendix A v hlavním dokumentu | 1 295 slov |
| Hlavní text + abstrakt + Appendix A | 6 837 slov |
| Hlavní PDF / Supporting Information PDF | 35 / 14 stran |

Počítá skript `analysis/audit_formal_requirements.py` z LaTeX AST pomocí Pandocu.
Vynechává nadpisy, popisky, tabulky, citace a vysazené rovnice. Každý inline
matematický výraz počítá jako jednu jednotku; slova se spojovníkem a rozsahy jako
jednu jednotku. Počty nejsou totožné s hrubým počtem všech slov Wordu.
Prohlášení za hlavním textem a literatura nejsou v uvedených součtech.

Hranice 250 slov pro abstrakt a pět klíčových slov byly použity jako konzervativní
přípravné cíle. **Nejsou zde vydávány za ověřené současné limity JTB.** Současný
maximální rozsah hlavního textu nebyl ověřen. Nalezený archiv oficiálního Author
Information Pack z 7. srpna 2014 není důkazem dnešních pravidel.

## Co zbývá před odesláním

1. V aktuálním JTB Guide for Authors nebo odesílacím portálu ověřit limit
   abstraktu a hlavního textu, počet klíčových slov, případný graphical abstract
   a další časopisecké položky. Přímý průvodce nešel načíst (HTTP 403 / chyba
   přístupu); jeho nedostupnost není důkazem absence těchto požadavků.
2. Vyřešit zpřístupnění nových odvozených dat a analytického kódu podle aktuální
   datové politiky JTB. Zdrojová anonymizovaná CT a anatomické podklady jsou
   veřejné v BoneDat. **Nové FE výsledky, predikce, load-mixture výstupy a kód
   v tomto repozitáři veřejně uložené nejsou.** Data Availability Statement tuto
   skutečnost přesně uvádí. DOI BoneDat nelze používat jako doklad uložení nových
   výsledků. Současná datová varianta přidělená JTB nebyla ověřena; nelze proto
   označit časopisecký požadavek na data za splněný.
3. Při odesílání vyplnit časopisecké formuláře, včetně požadovaného formuláře
   competing interests, pokud portál vyžaduje soubor ze svého nástroje. Připravený
   Word obsahuje prohlášení autorů, není výstupem oficiálního Declarations Tool.
   Potvrzení, že práce současně není posuzována jiným časopisem, nebylo v tomto
   rozhovoru získáno a nebylo doplněno jako skutečnost. Přítomnost konkrétního
   submission formuláře či povinnost cover letter ověřit v portálu.

## Podklady ověřené na webu

- [Současná AI politika Elsevier](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals): rozlišuje deklaraci při přípravě textu a podrobný popis v Methods při použití ve výzkumu; dovoluje reprodukovatelné vizualizace skutečných dat s odpovídajícím zveřejněním použití nástroje.
- [Elsevier Highlights](https://www.elsevier.com/researcher/author/tools-and-resources/highlights): samostatný editovatelný soubor, 3–5 bodů, do 85 znaků včetně mezer.
- [JTB na webu vydavatele](https://shop.elsevier.com/journals/journal-of-theoretical-biology/0022-5193): Highlights jsou vyžadovány.
- [Elsevier: formáty obrázků a popisky](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-formats-checklist): samostatné obrázky a popiskový soubor; PDF a Word patří mezi přijímané formáty.
- [Elsevier Research Data Guidelines](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines): požadavky na data se liší podle časopisecké varianty, proto zde není předpokládána konkrétní varianta JTB.
- [BoneDat, veřejný záznam zdrojových dat](https://zenodo.org/records/15189761): podklady pro stávající tvrzení o dostupnosti zdrojových dat.
- [Aktuální JTB Guide for Authors](https://www.sciencedirect.com/journal/journal-of-theoretical-biology/publish/guide-for-authors): nepodařilo se načíst; časopisecké limity zůstávají neověřené.

Podrobné strojové kontroly jsou v `formal_audit.json` a `export_validation.json`.
Původní soubory této formální revize jsou zachovány v `before/`.
