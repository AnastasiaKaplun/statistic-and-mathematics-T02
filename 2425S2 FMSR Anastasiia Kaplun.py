from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Compatible with browser-based GitHub Codespaces
import matplotlib.pyplot as plt
from scipy.stats import norm, t, sem, ttest_ind, f_oneway, beta
from sympy import symbols, exp, diff, solve, integrate

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset_2130683.csv'
FIG = ROOT / 'fmsr_figures'
FIG.mkdir(exist_ok=True)
ALPHA = 0.05

def heading(label):
    print('\n' + '=' * 70 + '\n' + label + '\n' + '=' * 70)

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180, bbox_inches='tight')
    plt.close()
    print('Chart:', FIG / name)

def summary(group, column):
    return group[column].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(6)

def describe_test(name, result):
    print(f'{name}: statistic={result.statistic:.6f}; p={result.pvalue:.8g}; '
          + ('reject H0' if result.pvalue < ALPHA else 'do not reject H0')
          + f' at alpha={ALPHA}')

# Q0 — DATA PREPARATION
heading('Q0: Data preparation and missing values')
raw = pd.read_csv(DATA)
required = ['ad_id', 'platform', 'impressions', 'engagements', 'reach', 'ctr', 'cvr']
missing_cols = set(required) - set(raw.columns)
if missing_cols:
    raise ValueError(f'Missing required columns: {missing_cols}')
print('Original rows:', len(raw))
print('Missing values by column:\n', raw.isna().sum().to_string())
print('Duplicate non-null ad IDs:', raw['ad_id'].dropna().duplicated().sum())
print('Duplicate full rows:', raw.duplicated().sum())
for col in ['impressions', 'engagements', 'reach', 'ctr', 'cvr']:
    raw[col] = pd.to_numeric(raw[col], errors='coerce')
# Retain rows with missing ad_id: the identifier is not needed for calculations.
# Retain rows with missing reach for all other analyses; drop reach NA locally.
data = raw.dropna(subset=['platform', 'impressions', 'engagements', 'ctr', 'cvr']).copy()
data = data[(data['impressions'] > 0) & (data['engagements'] >= 0)
            & data['ctr'].between(0, 1) & data['cvr'].between(0, 1)].copy()
data['engagement_rate'] = data['engagements'] / data['impressions']
print('Rows retained for non-reach analyses:', len(data))
print('Missing reach in retained rows:', data['reach'].isna().sum())
print('Zero-impression rows excluded from rate-based analyses:', (raw['impressions'] == 0).sum())
print('Engagement rates above 100%:', (data['engagement_rate'] > 1).sum())
print('Note: ID-missing rows retained; reach-missing rows excluded only for reach calculations.')

# Q1a — ADS PER PLATFORM
heading('Q1a: Number of ads by platform')
counts = raw['platform'].value_counts().sort_index()
print('Total ads:', len(raw), '\n', counts.to_string())
counts.plot.bar(figsize=(8, 5), title='Number of ads by platform')
plt.ylabel('Number of ads'); plt.xlabel('Platform'); plt.xticks(rotation=0)
save('q1a_ads_by_platform.png')

# Q1b — IMPRESSIONS
heading('Q1b: Impressions per ad and distribution')
imp = data['impressions'].dropna()
print(imp.describe().to_string())
print('Mean impressions:', imp.mean(), 'Median:', imp.median(), 'Skewness:', imp.skew())
plt.figure(figsize=(8, 5)); plt.hist(imp, bins=25, edgecolor='white')
plt.xlabel('Impressions per ad'); plt.ylabel('Frequency')
plt.title('Distribution of impressions per ad')
save('q1b_impressions_histogram.png')
print('Interpretation: inspect histogram and mean/median/skew before choosing a distribution. '
      'Normal is a candidate only if reasonably symmetric and bell-shaped; avoid asserting a formal fit without evidence.')

# Q1c — REACH BY PLATFORM
heading('Q1c: Reach by platform')
print(summary(raw.groupby('platform'), 'reach').to_string())
plt.figure(figsize=(8, 5))
platforms = sorted(data['platform'].unique())
plt.boxplot([raw.loc[raw.platform == p, 'reach'].dropna() for p in platforms], tick_labels=platforms)
plt.ylabel('Unique viewers (reach)'); plt.title('Reach per ad by platform')
save('q1c_reach_by_platform.png')

# Q2a — ENGAGEMENT RATE
heading('Q2a: Engagement rate = engagements / impressions')
print('Overall mean of ad-level engagement rates:', data['engagement_rate'].mean())
print('Overall impressions-weighted rate:', data['engagements'].sum() / data['impressions'].sum())
print(summary(data.groupby('platform'), 'engagement_rate').to_string())
print('Platform-weighted engagement rates:')
print(data.groupby('platform').apply(lambda g: g['engagements'].sum()/g['impressions'].sum(), include_groups=False).to_string())

# Q2b/c — SATURATION FUNCTION AND DERIVATIVE
heading('Q2b/c: Engagement saturation model and derivative')
a = float(data['engagement_rate'].max())
b = 0.000013
I = symbols('I', nonnegative=True)
E = a * (1 - exp(-b * I))
dE = diff(E, I)
print('Maximum observed engagement rate (a):', a)
print('E(I) =', E)
print("E'(I) =", dE)
print('Interpretation: derivative is positive but decreases toward zero as impressions grow.')
xgrid = np.linspace(0, data['impressions'].max(), 500)
ygrid = a * (1 - np.exp(-b * xgrid))
plt.figure(figsize=(8, 5)); plt.plot(xgrid, ygrid, label='E(I) = a(1-exp(-bI))')
plt.axhline(a, linestyle='--', label=f'Asymptote a={a:.3f}')
plt.xlabel('Impressions'); plt.ylabel('Modeled engagement rate')
plt.title('Exponential saturation of engagement rate'); plt.legend()
save('q2b_engagement_saturation.png')

# Q3a — LINKEDIN CTR VS OTHER PLATFORMS
heading('Q3a: Is LinkedIn CTR lower than the other platforms?')
print(summary(data.groupby('platform'), 'ctr').to_string())
ctr_groups = [data.loc[data.platform == p, 'ctr'].dropna().to_numpy() for p in platforms]
print('Omnibus one-way ANOVA, H0: equal platform CTR means:')
describe_test('ANOVA', f_oneway(*ctr_groups))
linkedin = data.loc[data.platform == 'LinkedIn', 'ctr'].dropna()
for p in platforms:
    if p != 'LinkedIn':
        other = data.loc[data.platform == p, 'ctr'].dropna()
        describe_test(f'LinkedIn < {p} (Welch independent-samples t-test)',
                      ttest_ind(linkedin, other, equal_var=False, alternative='less'))
print('Three pairwise tests: optional Bonferroni threshold 0.05/3 =', ALPHA / 3,
      '(not a class-taught method; report any use separately).')
plt.figure(figsize=(8, 5)); plt.boxplot(ctr_groups, tick_labels=platforms)
plt.title('CTR by platform'); plt.ylabel('Click-through rate')
save('q3a_ctr_by_platform.png')

# Q3b — LINKEDIN CVR > TWITTER CVR
heading('Q3b: Is LinkedIn CVR greater than Twitter CVR?')
print(summary(data.groupby('platform'), 'cvr').to_string())
li_cvr = data.loc[data.platform == 'LinkedIn', 'cvr'].dropna()
tw_cvr = data.loc[data.platform == 'Twitter', 'cvr'].dropna()
describe_test('LinkedIn > Twitter (Welch independent-samples t-test)',
              ttest_ind(li_cvr, tw_cvr, equal_var=False, alternative='greater'))

# Q3c — FACEBOOK CVR CONFIDENCE INTERVAL
heading('Q3c: 95% CI for mean Facebook ad CVR')
# Assignment says Facebook in introduction but 'all ads' in question wording.
# Primary interpretation: all Facebook ads (not all platforms).
fb = data.loc[data.platform == 'Facebook', 'cvr'].dropna()
fb_mean = fb.mean()
fb_ci = t.interval(confidence=0.95, df=len(fb)-1, loc=fb_mean, scale=sem(fb))
print(f'Facebook n={len(fb)}, sample mean={fb_mean:.8f}, 95% t CI={fb_ci}')
print('CI estimates the population mean Facebook-ad CVR, not individual ad CVRs.')
all_cvr = data['cvr'].dropna()
all_ci = t.interval(confidence=0.95, df=len(all_cvr)-1, loc=all_cvr.mean(), scale=sem(all_cvr))
print('Alternative interpretation, ALL platforms: mean=', all_cvr.mean(), '95% CI=', all_ci)

# Q3d — TOP DECILE UNDER ASSUMED NORMAL DISTRIBUTION
heading('Q3d: Threshold for top 10% Facebook CVR')
mu, sigma = 0.0812, 0.001
threshold = norm.ppf(0.90, loc=mu, scale=sigma)
print('90th percentile CVR:', threshold, f'({threshold*100:.4f}%)')

# Q4a/b — SUCCESS > 11% ENGAGEMENT RATE
heading('Q4a/b: Success probabilities by platform')
cutoff = 0.11
data['successful'] = data['engagement_rate'] > cutoff
success_count = int(data['successful'].sum())
print(f'Overall: {success_count}/{len(data)} = {success_count/len(data):.8f}')
print(data.groupby('platform')['successful'].agg(['sum', 'count', 'mean']).to_string())
ig_success = data.loc[data.platform == 'Instagram', 'successful'].mean()
fb_success = data.loc[data.platform == 'Facebook', 'successful'].mean()
print('Instagram success probability > Facebook?', ig_success > fb_success)
print('Difference (Instagram minus Facebook):', ig_success - fb_success)

# Q4c/d — BETA-BINOMIAL BAYESIAN UPDATE
heading('Q4c/d: Bayesian update with 2 successes in 50 new ads')
# Class method: alpha=last-year successes; beta=last-year failures.
# A beta prior with zero parameters is improper, so handle edge cases explicitly.
prior_alpha, prior_beta = success_count, len(data)-success_count
if prior_alpha == 0 or prior_beta == 0:
    print('Empirical Beta(successes, failures) prior is improper (zero shape parameter).')
    print('Using Beta(1,1) weak prior instead; this differs from empirical class convention.')
    prior_alpha, prior_beta = 1, 1
post_alpha, post_beta = prior_alpha + 2, prior_beta + 48
prior_mean = prior_alpha / (prior_alpha + prior_beta)
post_mean = post_alpha / (post_alpha + post_beta)
print('Prior Beta parameters:', prior_alpha, prior_beta, 'mean:', prior_mean)
print('Posterior Beta parameters:', post_alpha, post_beta, 'mean:', post_mean)
print('Posterior mean exceeds prior mean?', post_mean > prior_mean)
print('P(true success probability > 11%) =', 1-beta.cdf(cutoff, a=post_alpha, b=post_beta))
p_grid = np.linspace(0.0001, 0.9999, 500)
plt.figure(figsize=(8, 5))
plt.plot(p_grid, beta.pdf(p_grid, prior_alpha, prior_beta), label='Prior')
plt.plot(p_grid, beta.pdf(p_grid, post_alpha, post_beta), label='Posterior')
plt.axvline(cutoff, linestyle='--', label='11% threshold')
plt.xlabel('Probability of successful ad'); plt.ylabel('Probability density')
plt.title('Beta prior and posterior'); plt.legend()
save('q4c_bayesian_update.png')

# Q5a/b — AD FATIGUE AND ACCUMULATED IMPRESSIONS
heading('Q5a/b: Ad fatigue and total accumulated impressions')
t_day = symbols('t', real=True)
new_impressions = -50.64*t_day**2 + 1215.36*t_day
slope = diff(new_impressions, t_day)
peak_day = solve(slope, t_day)[0]
peak_daily = new_impressions.subs(t_day, peak_day)
accumulated = integrate(new_impressions, (t_day, 0, peak_day))
print('Daily new impressions:', new_impressions)
print('First derivative:', slope)
print('Day when new impressions peak:', float(peak_day))
print('Daily new impressions at peak:', float(peak_daily))
print('Cumulative impressions through peak day (continuous model):', float(accumulated))
print('Interpretation: fatigue begins when daily new impressions start declining after the peak.')
days = np.linspace(0, 24, 300)
values = -50.64*days**2 + 1215.36*days
plt.figure(figsize=(8, 5)); plt.plot(days, values, label='New impressions per day')
plt.axvline(float(peak_day), linestyle='--', label=f'Peak day {float(peak_day):.2f}')
plt.xlabel('Days after launch'); plt.ylabel('New impressions per day')
plt.title('Ad fatigue: daily new impressions'); plt.legend()
save('q5a_ad_fatigue.png')
