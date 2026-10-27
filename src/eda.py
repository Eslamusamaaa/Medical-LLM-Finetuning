"""Exploratory data analysis of the Arabic medical QA dataset."""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %%
from huggingface_hub import hf_hub_download
dataset = load_dataset("madilcy/arabic-medical-qa-MERGED-MAQA-MMMLU-MI")
df = pd.DataFrame(dataset['train'])
df.head()

# %%
print("SCHEMA OVERVIEW")
print(df.dtypes)
print()
df.info()
print(f"Columns    : {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
duplicates = df.duplicated().sum()

# %%
df = df.dropna(subset=['reply'])

print(f"Rows after cleaning : {df.shape[0]}")
print(f"Dropped             : 101 rows")
print(f"Duplicates          : {duplicates} rows")

# %%
col_q = df.columns[0]
col_a = df.columns[1]
df['question_word_count'] = df[col_q].apply(lambda x: len(str(x).split()))
df['answer_word_count'] = df[col_a].apply(lambda x: len(str(x).split()))

print("\n Word Count Statistics ")
display(df[['question_word_count', 'answer_word_count']].describe())

# %%
df['question_len'] = df['question'].astype(str).apply(len)

plt.figure()
sns.histplot(df['question_len'], kde=True)
plt.title("Question Length Distribution")
plt.show()

# %%
df['reply_len'] = df['reply'].astype(str).apply(len)

plt.figure()
sns.histplot(df['reply_len'], kde=True)
plt.title("Reply Length Distribution")
plt.show()

# %%
plt.figure()
sns.scatterplot(x=df['question_len'], y=df['reply_len'])
plt.title("Question vs Reply Length")
plt.xlabel("Question Length")
plt.ylabel("Reply Length")
plt.show()

# %%
def fix_arabic(text):
    reshaped = arabic_reshaper.reshape(str(text))
    bidi_text = get_display(reshaped)
    return bidi_text
df['category_fixed'] = df['category'].apply(fix_arabic)

sns.countplot(x='category_fixed', data=df)
plt.xticks(rotation=45)
plt.show()

# %%
corr = df[['q_body_count', 'a_body_count', 'question_len', 'reply_len']].corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %%
print(df[['q_body_count','a_body_count']].describe())
print(df[['q_body_count','a_body_count']].isnull().sum())

# %%
df = df.drop(columns=['q_body_count', 'a_body_count'])

# %%
def fix_arabic(text):
    reshaped = arabic_reshaper.reshape(str(text))
    bidi_text = get_display(reshaped)
    return bidi_text
words_fixed = [fix_arabic(w) for w in words]

plt.figure()
plt.bar(words_fixed, counts)
plt.title(fix_arabic("Top 10 Most Common Words in Questions"))
plt.xticks(rotation=45)
plt.show()

# %%
all_words_ans = ' '.join(df['reply']).split()
all_words_ans = [w for w in all_words_ans if len(w) > 2]

top_words_ans = Counter(all_words_ans).most_common(10)
words_ans, counts_ans = zip(*top_words_ans)

visual_words_ans = [process_arabic_text_for_visuals(w) for w in words_ans]

plt.figure(figsize=(12, 5))
plt.bar(visual_words_ans, counts_ans, color='coral') 

plt.title('Top 10 Most Common Words in Answers')
plt.xlabel('Word')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../reports/top_answer_words.png', dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(y=df['question_len'], ax=axes[0], color='steelblue',
            flierprops=dict(marker='o', markerfacecolor='steelblue', markersize=3, alpha=0.4))
axes[0].set_title('Question Length Distribution', fontsize=14)
axes[0].set_ylabel('Character Length')

sns.boxplot(y=df['reply_len'], ax=axes[1], color='coral',
            flierprops=dict(marker='o', markerfacecolor='coral', markersize=3, alpha=0.4))
axes[1].set_title('Reply Length Distribution', fontsize=14)
axes[1].set_ylabel('Character Length')

plt.suptitle('Box Plots: Character Length Distributions', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%
def fix_arabic(text):
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

top_cats = df['category'].value_counts().nlargest(6).index
df_top = df[df['category'].isin(top_cats)].copy()
df_top['category_fixed'] = df_top['category'].apply(fix_arabic)

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_top, x='category_fixed', y='question_len', palette='Set2',
            flierprops=dict(marker='o', markersize=3, alpha=0.4))
plt.title('Question Length by Category (Top 6)', fontsize=15)
plt.xlabel('Category')
plt.ylabel('Question Length (chars)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# %%
melted = df[['question_len', 'reply_len']].melt(
    var_name='Type', value_name='Character Length'
)
melted['Type'] = melted['Type'].map({
    'question_len': 'Question',
    'reply_len': 'Answer'
})

plt.figure(figsize=(10, 6))
sns.violinplot(data=melted, x='Type', y='Character Length',
               palette=['steelblue', 'coral'], inner='quartile')
plt.title('Violin Plot: Character Length — Questions vs Answers', fontsize=15)
plt.ylabel('Character Length')
plt.xlabel('')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(11, 5))
sns.kdeplot(df['question_len'], label='Question Length', fill=True, alpha=0.4, color='steelblue')
sns.kdeplot(df['reply_len'], label='Reply Length', fill=True, alpha=0.4, color='coral')
plt.title('KDE: Question vs Reply Length Overlap', fontsize=15)
plt.xlabel('Character Length')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# %%
def fix_arabic(text):
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

cat_counts = df['category'].value_counts()
top_n = 6
top_cats_pie = cat_counts.nlargest(top_n)
other_count = cat_counts.iloc[top_n:].sum()
if other_count > 0:
    top_cats_pie['Other'] = other_count

labels = [fix_arabic(l) for l in top_cats_pie.index]

plt.figure(figsize=(9, 9))
plt.pie(
    top_cats_pie.values,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('pastel', len(top_cats_pie))
)
plt.title('Category Distribution (Top 6 + Other)', fontsize=15)
plt.tight_layout()
plt.show()

# %%
numeric_cols = ['question_len', 'reply_len', 'question_word_count', 'answer_word_count']
pair_df = df[numeric_cols].copy()
pair_df.columns = ['Q Len', 'A Len', 'Q Words', 'A Words']

g = sns.pairplot(pair_df, diag_kind='kde', plot_kws={'alpha': 0.3, 'color': 'steelblue'})
g.fig.suptitle('Pair Plot: Numeric EDA Features', y=1.02, fontsize=15)
plt.tight_layout()
plt.show()

# %%
def fix_arabic(text):
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

top_cats_bar = df['category'].value_counts().nlargest(6).index
df_bar = df[df['category'].isin(top_cats_bar)].copy()
df_bar['category_fixed'] = df_bar['category'].apply(fix_arabic)

plt.figure(figsize=(13, 6))
sns.barplot(
    data=df_bar,
    x='category_fixed',
    y='reply_len',
    palette='muted',
    capsize=0.1,
    errwidth=1.5
)
plt.title('Mean Reply Length by Category (Top 6)', fontsize=15)
plt.xlabel('Category')
plt.ylabel('Mean Reply Length (chars)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(11, 5))
sns.ecdfplot(df['question_len'], label='Question Length (chars)', color='steelblue')
sns.ecdfplot(df['reply_len'], label='Reply Length (chars)', color='coral')
plt.title('ECDF: Character Length Cumulative Distribution', fontsize=15)
plt.xlabel('Character Length')
plt.ylabel('Proportion')
plt.legend()
plt.tight_layout()
plt.show()

