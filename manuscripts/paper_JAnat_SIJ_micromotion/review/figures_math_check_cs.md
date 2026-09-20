# Kontrola obrázků a matematického zápisu — 20. 9. 2026

Vizuálně zkontrolováno všech šest hlavních obrázků a obrázek S1 před i po úpravách, včetně sazby PDF.

- Obr. 1: nové kompaktní uspořádání; odstraněny přeplněné popisky a schematické morfometrické překryvy. Centroid ROI již není označen jako fyzický kontaktní bod; komponenty nejsou zaměňovány za měřený smyk či rozevírání.
- Obr. 2: asymetrie přesně jako absolutní rozdíl norem; matematický dolní index ML rotace.
- Obr. 3: číselné popisky posunuty od intervalů nejistoty.
- Obr. 4 a 6: menší nadbytečné vertikální rozestupy.
- Obr. 5: sdílené řádkové popisky a konzistentní označení poměru rozptylů.
- S1: pět řádků a dva sloupce pro čitelnost na portrétní stránce; odstraněny statistické rámečky zakrývající body.
- Sazba: upravena pravidla floatů, aby nevytvářela zbytečné mezery mezi obrázky.

Matematická příloha doplněna o definice symbolů, objemový pullback a izotropní konstitutivní vztah, přesnou vazbu bodového rozdílu posunů na relativní rigidní transformaci, výběr ROI, referenční jednotky logaritmů, HC3 kovarianci, význam allometrických exponentů a estimand znaménkového testu. Geometrická nezatížená reference je výslovně odlišena od samostatně vyřešené rovnováhy při předpětí. Opraven odkaz na podmíněné křivky v S1.

Validace: 16 testů v test_scientific_audit_math.py a test_sij_reference_configuration.py prošlo. První soubor zahrnuje i širší algebraické kontroly repozitáře; nepředstavuje validaci všech rovnic článku. Rovnice kinematiky a statistické definice byly navíc porovnány s implementací. Generátor S1 ověřuje shodu sklonů s tabulkami numerickými assertions. Obě PDF sestavena; bez undefined references, overfull boxes a LaTeX warnings. Číselné výsledky ani FE řešení se neměnily. Tato kontrola není experimentální validací modelu.
