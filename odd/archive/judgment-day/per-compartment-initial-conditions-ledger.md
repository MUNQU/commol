# Judgment Day Ledger — per-compartment initial conditions

- Target: `feature/per-compartment-initial-conditions` @ `5f0bf896b097b1291898e6ea086ac83aa747429a` vs base `0fa4e516fc423dd4e7735b5ae190b8c090471a53` (23 files, ~2458 insertions)
- Round 1 judges: `jd-judge-a` (task muv3b3u6-2-vzbr), `jd-judge-b` (task muv3b3u6-3-mcgd), blind, parallel, read-only
- Skill resolution: paths-injected (`/Users/david/.agents/skills/judgment-day/SKILL.md`)

## Confirmed severe (both judges) — fixable

None.

## Suspect (one judge only — no auto-fix)

- **JD-A-001** (CRITICA, jd-judge-a) `py-commol/commol/context/model.py:610-620` —
  **DOWNGRADED a WARNING tras corroboración cruzada (PARCIAL).** El síntoma
  del informe se CONFIRMÓ (`bin_fractions` queda en 1.0 y el informe dice
  `calibrated: true` con la fracción vieja), pero la afirmación central se
  REFUTÓ: la desviación cubre todos los compartimentos afectados
  (`diverted.update(covered)`), el reload JSON reproduce el engine
  bit-idénticamente y el doble apply es idempotente; el valor ajustado se
  conserva íntegro vía overrides. El defecto observable es solo la fila
  agregada engañosa del informe → se fusiona con F1. Nota de diseño: persistir
  también la fracción agregada exige reescalar los bins hermanos (validador
  suma-1) y es innecesario; se mantiene la desviación exclusiva.
  Hallazgo incidental: `update_bin_fractions` no re-ejecuta el validador de
  suma 1 (marcado como FIX-INC-1).

## Confirmed INFO (ambos jueces, mismo hallazgo)

- **F1** (WARNING, A2 + B1) `py-commol/commol/api/reporting.py:285-317` —
  Con overrides declarados, las filas bin/categoría del informe de
  condiciones iniciales reportan la expansión uniforme y sobrestiman la
  población efectiva del engine; `subgroup_population` solo integra el
  override cuando la cadena de categorías selecciona exactamente un
  compartimento.

## INFO (un solo juez, no severo)

- **A3** (WARNING) `model.py:547-549` — orden de JSON no determinista (set al desviar overrides), verificado con PYTHONHASHSEED.
- **A4** (WARNING) `model.py:357-368` — `apply_calibration_parameters` no atómico: validación posterior muta `bin_fractions` antes de fallar.
- **A5** (WARNING) `reporting.py:238-259` — `_override_is_calibrated` O(n³) (512 compartimentos → 4.66 s), medido con cProfile.
- **A6** (WARNING) `docs/guide/calibration.md:834-840`, `AGENT.md:878-881,483-489` — documentación obsoleta: aún dice que los IC por compartimento no se pueden escribir de vuelta.
- **A7** (WARNING) `builder.rs:442-451` — Rust no valida rango ni duplicados de overrides; JSON crudo admite fracciones negativas/duplicadas (población negativa posible).
- **B2** (SUGGESTION) `py-commol/src/core.rs:195-231` — binding PyO3 hardcodea `compartment_fractions: Vec::new()`; `.pyi` sin el campo.
- **A8** (SUGGESTION) `model.py:622-628` — divergencia replay Python vs engine con categoría compartida por dos estratificaciones.
- **A9** (SUGGESTION) `model.py:547` — `population_size: 0` → `ZeroDivisionError` sin error de dominio.
- **B3** (SUGGESTION) `initial_conditions.py:144-151` — validador de unicidad O(N²).

## Contradicciones

Ninguna.

## Corroboración cruzada (ronda 1)

Veredictos del agente independiente (gentle-ai-verify, task muv4agnb-4-g2ch):
JD-A-001 PARCIAL (informe CONFIRMADO, pérdida de valor REFUTADA, degradada),
F1/A3/A4/A6/A7/A8/A9/B2/B3 CONFIRMADOS, A5 CONFIRMADO con corrección
(el disparador es un id de **categoría** calibrada; un bin calibrado
cortocircuita). B2 verificado estáticamente. Sin mutaciones del repositorio.

## Recuento ronda 1

confirmados severos: 0 · sospechosos: 1 (JD-A-001, degradada a WARNING) ·
confirmados INFO: 1 (F1) · INFO adicionales: 9 · contradicciones: 0 ·
incidental: FIX-INC-1 (`update_bin_fractions` sin validador suma-1)

## Ronda 1: corrección autorizada por el usuario (todos los confirmados)

- FIX-1 (F1 + JD-A-001): informe con población efectiva (expansión + overrides);
  `subgroup_population` integra overrides también con cadenas de >1
  compartimento, verificando los llamadores existentes.
- FIX-2 (A3, A4, A9, FIX-INC-1): orden determinista al desviar (ordenar
  `diverted`); atomicidad de `apply_calibration_parameters` (rollback o
  validación previa); `population_size > 0` como error de dominio; validador
  suma-1 en `update_bin_fractions`.
- FIX-3 (A5, B3): memoización de `_category_pairs` en el camino del informe;
  unicidad O(N) en `validate_compartment_fractions_unique`.
- FIX-4 (A7, B2): validación Rust en `initialize_population` (rango 0..=1,
  duplicados → Err) con tests; exponer `compartment_fractions` en el binding
  PyO3 `InitialConditions` y en `_commol_rs.pyi` (incluye `bin_fraction`
  mal nombrado); `maturin develop` tras el cambio.
- FIX-5 (A6): actualizar `docs/guide/calibration.md` (admonition y tabla de
  routing), `AGENT.md` (878-881, 483-489) y guía de usuario para
  `compartment_fractions`/`set_initial_conditions`; nota de racional en el
  design doc (por qué la desviación no persiste la fracción agregada).
- FIX-6 (A8): replay de categorías con instantánea por parámetro (paridad con
  el engine `calibration_problem.rs:1077-1100`) y comportamiento documentado
  para nombres de categoría compartidos entre estratificaciones.

## Estado

- [x] Ronda 1: juicio dual + ledger + corroboración
- [x] Ronda 1: corrección acotada (FIX-1..FIX-6) — jd-fix-agent task muv5zlke-5-jesu; 12/12 IDs
      corregidos; delta 859+/114- en 12 archivos SIN commitear; puerta final verde:
      pytest 550 passed, cargo 90+25 passed, fmt/clippy/ruff/ty limpios.
- [x] Re-juicio acotado (ledger + delta) — jd-judge-a (relanzado tras timeout, task
      muv8midc-8-pnjs) y jd-judge-b (task muv6obn5-7-e5qz). B: JD-F-1 CRITICA
      (doble apply no idempotente con categorías compartidas, sonda determinista
      900->860) + JD-F-2 WARNING (guard total<=1.0 rechaza valores que el engine
      acepta). A: sin CRITICAs; 4 WARNINGs (fracción con denominadores mezclados
      en el informe, bin sin calibrar pierde el marcador None, stub .pyi con
      atributos fantasma, doc de guía con semántica antigua de fraction) y 1
      SUGGESTION (changelog sobreafirma la validación Rust). Cobertura: A no probó
      el caso categoría-compartida del doble apply; sin contradicción factual.
      Los 12 IDs de la ronda 1 verificados como corregidos por ambos jueces.
- [x] Corroboración cruzada de los hallazgos del re-juicio (gentle-ai-verify task
      muv9cacc-9-gv7a): 7/7 CONFIRMADOS. JD-F-1 reproducida (900->860) y mecanismo
      identificado: cada compartimento participa en dos pares ordenados y el
      segundo apply parte de una base que ya incluye los overrides del primero;
      la no-idempotencia está confinada a la rama de categoría compartida
      (model.py:691). JD-F-2: el espejo exacto de calibration_problem.rs:1077-1100
      da total 1100 y set_initial_condition no tiene check de total.
- [x] Decisión: ronda 2 (final) autorizada por la instrucción permanente del
      usuario (corregir todo lo confirmado tras corroboración agente-a-agente).
- [x] Ronda 2: corrección acotada (jd-fix-agent task muva84uc-a-9pu7) — JD-F-1 fija
      (replay desde base declarada: doble apply byte-idéntico y misma población),
      JD-F-2 fija (write-back con validate_total=False; guards públicos intactos),
      A-1 fija (denominador único population_size), A-2 fija (marcador None
      preservado), A-3 fija (getters PyO3 reales + stub honesto), A-4 fija (docs),
      A-5 PARCIAL: (a) validador de construcción Pydantic + changelog veraz;
      (b) no aplicado por conflicto directo con JD-F-2 (el engine debe aceptar
      estados no conservadores del replay; se preserva la severidad mayor).
      Puerta final: pytest 559 passed, cargo 90+25 passed, fmt/clippy/ruff/ty limpios.
- [x] Re-juicio 2 (final) — jd-judge-a (muvbdcrn-b-jng8) + jd-judge-b (muvbdcro-c-mpxt).
      Confirmada por AMBOS jueces: CRITICA de contradicción escritura/lectura — el
      write-back persiste estados no conservadores (sum 1.1) que la nueva guarda de
      construcción Pydantic (A-5a) rechaza al recargar; Model.from_json/
      model_validate_json de model_dump_json() lanza ValidationError (B ejecutó la
      sonda). Solo juez B: JD-G-1 CRITICA sospechosa (el replay desde base declarada
      descarta overrides preexistentes; reload ya no reproduce el estado calibrado;
      sondas deterministas [100,200,200,300] vs [200,200,200,300]); JD-G-3 WARNING
      (update_bin_fractions rechaza el write-back no conservador de un bin). Solo
      juez A: WARNING intervalo en unidad equivocada para categorías condicionales;
      WARNING deriva de segundo apply con dicts mixtos (mecanística).
- [x] Verificación final independiente (gentle-ai-verify task muvbrq1q-d-sxjx):
      7/7 PASS — pytest 559 passed; cargo 90+25 passed; fmt/clippy limpios;
      ruff format/check limpios; ty limpio; sin commits (HEAD 5f0bf89, 12 archivos
      modificados + odd/judgment-day/ sin seguimiento); delta final +1204/-138.
- [x] VEREDICTO TERMINAL: ESCALATED — presupuesto agotado (2 rondas de corrección,
      2 re-juicios) con una CRITICA confirmada por ambos jueces pendiente. Los
      defectos residuales quedan documentados para decisión humana; ninguna ronda
      adicional está permitida.

## Residuales escalados (trabajo futuro, fuera del presupuesto JD)

1. ESC-1 (CRITICA, ambos jueces): decide la política de estados no conservadores —
   o la guarda de construcción Pydantic admite lo que el write-back persiste
   (reload de estados no conservadores), o el write-back vuelve a rechazarlos;
   ambas direcciones son coherentes, la mezcla actual no. Cubre también JD-G-3.
2. ESC-2 (CRITICA sospechosa, juez B, sondas deterministas): base del replay —
   live (paridad con el engine, no idempotente) vs declarada (idempotente, pierde
   overrides preexistentes). Requiere decisión de diseño y tests de paridad con
   overrides preexistentes y dicts mixtos.
3. ESC-3 (WARNING, juez A): intervalo del informe en unidad equivocada para
   categorías condicionales tras el cambio de denominador.
4. ESC-4 (SUGGESTION, juez A, no fila): asimetría del binding PyO3 — el
   constructor acepta bin_fractions Vec<(String, f64)> pero el getter devuelve
   Option<f64>; un bin None no puede round-trip.
