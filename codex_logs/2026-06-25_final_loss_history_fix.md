Updated FOM and ROM optimization runners so the final saved loss is recomputed at `result.u`, while evaluation history remains callback-only and excludes the extra final-loss check.
