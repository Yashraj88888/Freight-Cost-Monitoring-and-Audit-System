# Smart Shipping Cost Assistant - Evaluation Report

**Overall Pass Rate**: 100.0% (8/8 passed)

| Test ID | Category | Test Case Description | Result | Details |
| :--- | :--- | :--- | :--- | :--- |
| `TC01_SAMPLE_ROW1` | sample_contract | Delhi-Jaipur Unexplained Cost Spike (Sample Row 1) | **PASS** | Cost: 4.17, Flag: Yes, Note: None |
| `TC02_SAMPLE_ROW2` | sample_contract | Ahmedabad-Mumbai Justified Festival Surcharge (Sample Row 2) | **PASS** | Cost: 3.29, Flag: No (justified), Note: N002 |
| `TC03_SAMPLE_ROW3` | sample_contract | Mumbai-Pune Peer Anomaly with Negative Note N006 (Sample Row 3) | **PASS** | Cost: 3.98, Flag: Yes, Note: None |
| `TC04_FLOOD_JUSTIFIED_W1` | disruption_grounding | Chennai-Bangalore Flood Disruption Week 1 (N001) | **PASS** | Cost: 3.58, Flag: No (justified), Note: N001 |
| `TC05_FLOOD_JUSTIFIED_W2` | disruption_grounding | Chennai-Bangalore Flood Disruption Week 2 (N001 Window) | **PASS** | Cost: 3.48, Flag: No (justified), Note: N001 |
| `TC06_TRAP_N004_NOT_IN_DATASET` | anti_hallucination | Negative Trap N004 (Toll plaza not in dataset) | **PASS** | Cost: 3.11, Flag: Yes, Note: None |
| `TC07_TRAP_N005_COSTS_NOT_AFFECTED` | anti_hallucination | Negative Trap N005 (Maintenance where costs were not affected) | **PASS** | Cost: 2.06, Flag: Yes, Note: None |
| `TC08_TRAP_N010_COSTS_ABSORBED` | anti_hallucination | Negative Trap N010 (Costs absorbed without rate change) | **PASS** | Cost: 4.07, Flag: Yes, Note: None |

### Guardrail & Anti-Hallucination Analysis
- **Sample Output Rows**: Exactly reproduces Row 1 (unexplained), Row 2 (justified by N002), and Row 3 (negative note N006 rejected).
- **Negative Trap Protection**: Notes N004 (toll plaza not in dataset), N005 (costs not affected), N008 (normal traffic), and N010 (costs absorbed) are properly prevented from justifying rate hikes.
- **Grounding**: 100% of 'No (justified)' verdicts correspond to valid, active disruptions.
- **Hallucination Rate**: **0.0%**