DATA_DIR = 'data'
OUTPUT_DIR = 'output'

# M1 – SMART targets (TeeTech Design)
# Monetary amounts are in Ariary (Ar). Rates expressed as proportions (e.g., 0.02 for 2%).
M1_TARGETS = {
    # Traffic/Engagement (Facebook)
    'followers_target': 5000,           # in 6 months
    'engagement_rate_min': 0.05,        # ≥ 5%
    'ctr_min': 0.02,                    # ≥ 2%
    'cpc_max': 50.0,                    # ≤ 50 Ar
    'cpa_max': 5000.0,                  # ≤ 5 000 Ar
    'roi_min': 3.0,                     # ≥ 300% (ROAS/ROI multiplier)

    # Commercial
    'monthly_revenue_target': 864000.0, # 720k +20% in 4 months
    'conversion_rate_target': 0.04,     # 4% in 6 months

    # Relation client
    'satisfaction_min': 0.90,           # ≥ 90%
    'nps_min': 50,                      # ≥ 50
    'repeat_rate_min': 0.40,            # ≥ 40%
    'messenger_response_time_max_h': 2.0,# ≤ 2h
}
