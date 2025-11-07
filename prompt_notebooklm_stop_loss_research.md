# PROMPT DI RICERCA - NOTEBOOKLM

## Contesto del Problema

Ho un sistema di trading algoritmico basato su FreqTrade con strategia Ichimoku che presenta un problema critico di stop loss. Attualmente il bot ha queste performance:

**Statistiche Bot 8092:**
- **Trade totali analizzati:** 82
- **Profitto totale:** -6.36% (in perdita)
- **Trade vincenti:** 77 (93.9%)
- **Trade perdenti:** 5 (6.1%)

**Il Problema Fondamentale:**
Sebbene il tasso di vincita sia del 93.9%, il sistema è in perdita a causa di **3 trade con stop loss al -10%** che hanno annullato tutti i profitti generati dagli altri 79 trade vincenti:

1. **AVAX/USDT:** Perdita -10.17% (stop_loss)
2. **BNB/USDT:** Perdita -10.30% (stop_loss)
3. **DOT/USDT:** Perdita -10.13% (stop_loss)

**Configurazione Attuale:**
- Stop loss nativo FreqTrade: -10% (impostato come "extreme loss" hard limit)
- La strategia usa `custom_stoploss` ma spesso ritorna `None` per usare il nativo
- Itrade con perdite tra -2% e -6% sono gestiti da trailing stop loss
- Solo le perdite catastrofiche attivano lo stop loss al 10%

## Domande di Ricerca

### 1. Analisi del Problema
Analizza i dati sopra e rispondi:
- Perché uno stop loss al 10% è inefficace come rischio management?
- Qual è l'impatto psicologico ed economico di perdite del 10% singole?
- Come si confronta questo approccio con strategie di risk management professionali?

### 2. Best Practices di Stop Loss
Ricerca e documenta:
- Quali sono i livelli di stop loss considerati "industry standard" per trading intraday e swing trading?
- Come funzionano i sistemi di stop loss dinamici/adiattivi?
- Quali sono le alternative allo stop loss fisso al 10%?
- Qual è il ruolo del trailing stop loss e come dovrebbe essere configurato?

### 3. Strategie Ichimoku e Risk Management
Approfondisci:
- Come integrare correttamente i livelli di Ichimoku (Tenkan-sen, Kijun-sen) come stop loss dinamici?
- Quali sono le best practice per combinare indicatori tecnici con stop loss adattivi?
- Come gestire il rischio in strategie con alto tasso di vincita ma occasionali perdite massime?

### 4. Soluzioni Specifiche per FreqTrade
Ricerca:
- Come implementare stop loss basati su ATR (Average True Range) in FreqTrade?
- Qual è il modo corretto di usare `custom_stoploss` per sostituire completamente il nativo?
- Come implementare stop loss multi-livello (breakeven, parziale, totale) in FreqTrade?

### 5. Analisi Quantitativa
Valuta:
- Qual dovrebbe essere il rapporto risk/reward ottimale per questo tipo di strategia?
- Come calcolare il maximum drawdown accettabile dato il tasso di vincita del 93.9%?
- Quali metriche usare per valutare l'efficienza del sistema di stop loss?

## Obiettivo della Ricerca

Identificare un sistema di stop loss che:
1. **Protegga dai drawdown catastrofici** senza annullare i profitti
2. **Si adatti** alle condizioni di mercato e volatilità
3. **Mantenga** l'alto tasso di vincita della strategia
4. **Sia implementabile** in FreqTrade con la strategia Ichimoku esistente

## Output Atteso

Una strategia completa di risk management che includa:
- Configurazione specifica dei parametri di stop loss
- Codice di esempio per FreqTrade
- Piano di implementazione graduale
- Metriche di valutazione della performance

---
*Nota: Ho fornito dati reali dal mio sistema di trading. La ricerca dovrebbe basarsi su evidenze quantitative e best practices professionali del settore.*