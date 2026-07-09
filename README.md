# Mundial 2026 Predictor

**🇪🇸 Español** · [🇬🇧 English](README.en.md)

Modelo probabilístico de predicción de partidos del Mundial 2026 (fase
eliminatoria), construido con un enfoque de Risk Analytics: calibración,
backtesting sin fuga temporal y comparación honesta contra el mercado de
apuestas como benchmark.

**▶️ Correr localmente** — ver la sección "Cómo correrlo" más abajo.

## Qué hace

Estima, para cada cruce de la fase eliminatoria, la probabilidad de que cada
selección avance, y simula el cuadro restante para obtener la probabilidad de
título de cada equipo. Las predicciones se comparan contra la línea de cierre
de Pinnacle (la casa de apuestas más precisa) usando Brier score y log-loss.

El objetivo no es "ganarle al mercado", sino construir un modelo propio,
interpretable, que **compita de igual a igual** con el mercado y cuyos límites
estén medidos y documentados.

## Resultado principal (backtest justo, a 90 minutos)

Comparación manzanas con manzanas sobre el resultado a 90' (1X2), que es lo que
el mercado efectivamente cotiza:

| | Brier score | Log-loss |
|---|---|---|
| **Modelo** | 0.3915 | 0.7076 |
| **Mercado (Pinnacle)** | 0.4089 | 0.7261 |

Sobre 24 partidos de eliminatoria, el modelo fue más preciso que el mercado en
15. **Lectura honesta:** es una muestra chica (torneo en curso), así que esto
es una señal temprana de que el modelo es **competitivo / a la par de Pinnacle**,
no una prueba de ventaja sostenida. No es un sistema para ganarle a las casas.

## Cómo funciona (arquitectura por capas)

- **Datos** (`src/data.py`): descarga y valida el histórico de partidos
  internacionales (dataset martj42, ~49.500 partidos desde 1872).
- **Elo** (`src/elo.py`): rating de fuerza calculado **sin fuga temporal** (cada
  partido usa solo el rating previo). K variable por importancia (Mundial >
  eliminatorias > amistosos) y ventaja de local que se apaga en cancha neutral.
- **Modelo de goles** (`src/model.py`): Poisson bivariado (estilo Dixon-Coles)
  con las tasas de gol dependientes del Elo, ponderado por **recencia**
  (time-decay, vida media de 24 meses) para que pese el fútbol actual.
- **Eliminatoria** (`src/knockout.py`): convierte el resultado a 90' en
  probabilidad de avance, modelando alargue y penales (moneda 50/50).
- **Blend** (`src/blend.py`): combina el modelo Poisson con el Elo puro
  (ensemble). Corrige un sesgo del Poisson que subestimaba a los favoritos
  claros (ver más abajo).
- **Simulación** (`src/simulation.py`): Monte-Carlo del cuadro restante (20.000
  torneos) para las probabilidades de título.
- **Mercado** (`src/odds.py`): baja las cuotas de OddsPapi, les quita el margen
  (vig) y obtiene la probabilidad implícita de Pinnacle. Con caché local para
  no gastar la cuota gratuita.
- **Evaluación** (`src/evaluation.py`): backtest modelo vs. mercado con Brier y
  log-loss, sin fuga temporal (el modelo se reajusta a la fecha de cada partido).

## Un hallazgo del proceso

Durante el desarrollo, el modelo daba a Marruecos favorito sobre Francia pese a
que Francia tenía 150 puntos más de Elo. Comparando contra el mercado (Pinnacle
daba Francia 60%), se confirmó que el modelo de goles **sobre-reaccionaba a la
forma reciente** y subestimaba la diferencia de nivel. Se corrigió con una capa
de blend Elo+Poisson, calibrada contra los resultados reales. Este ciclo
—desconfiar de un resultado, testearlo contra un benchmark y corregir el sesgo—
es el núcleo metodológico del proyecto.

## Decisiones de diseño

- **Elo en vez de ranking FIFA.** El Elo se actualiza partido a partido y es
  mejor predictor; el ranking FIFA se actualiza por ventanas (quedó congelado
  antes del torneo) y fue diseñado para sembrar sorteos, no para predecir.
- **Poisson / boosting en vez de deep learning.** Con datos tabulares de bajo
  volumen, los modelos de goles interpretables rinden igual o mejor que una red
  neuronal, con más transparencia.
- **Blend calibrado con criterio (0.5), no al óptimo de muestra chica.** La
  calibración sobre 24 partidos favorecía Elo puro, pero se eligió un blend
  balanceado para no sobreajustar a un resultado frágil.

## Cómo correrlo

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app lee datos pre-computados de `data/app/*.json` y **nunca** llama a la API,
así que corre sin configuración. Para refrescar con los últimos resultados y
cuotas:

```bash
python scripts/update_data.py
```

Ese script es el único que llama a OddsPapi (vía `src/odds.py`, con caché en
`data/odds_cache/` para cuidar la cuota gratuita). Requiere `ODDSPAPI_API_KEY`
en un archivo `.env` (ver `.env.example`). Se corre después de cada jornada y
se commitean los `data/app/*.json` actualizados.

## Automatización

Un workflow de GitHub Actions (`.github/workflows/update-data.yml`) corre
`scripts/update_data.py` una vez por día (06:00 UTC) y commitea los
`data/app/*.json` resultantes de vuelta a `main` si hubo cambios. También se
puede disparar a mano desde la pestaña *Actions* (`workflow_dispatch`). La API
key se lee desde el secret `ODDSPAPI_API_KEY` del repositorio, nunca
hardcodeada. Si el script falla (p. ej. se agotó la cuota de la API), el
workflow falla y no commitea nada.

## Limitaciones y mejoras futuras

- **Muestra chica:** el backtest es sobre la eliminatoria en curso (~24
  partidos). Los resultados son señales tempranas, no evidencia de ventaja a
  largo plazo.
- **Bracket hardcodeado:** la estructura del cuadro se actualiza a mano cada
  ronda; una versión más robusta la derivaría automáticamente de los resultados.
- **Solo datos de equipo:** no incorpora valor de plantel (Transfermarkt) ni
  datos a nivel jugador, que la literatura sugiere que mejoran la predicción.

## Stack

Python · pandas · numpy · scipy · statsmodels · scikit-learn · Streamlit ·
OddsPapi API · Elo · Poisson bivariado (Dixon-Coles) · Monte-Carlo

## Datos

- [International football results 1872–2026 (martj42)](https://github.com/martj42/international_results)
- Cuotas: [OddsPapi](https://oddspapi.io/) (línea de cierre de Pinnacle)

---

*Nota: las apuestas conllevan riesgo financiero real. Este es un proyecto de
modelado y portfolio, no asesoramiento para apostar.*
