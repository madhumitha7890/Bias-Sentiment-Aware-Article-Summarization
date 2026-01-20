import pandas as pd
import random
import faker

# Initialize faker for realistic-looking text
fake = faker.Faker()

# Function to create a single synthetic article with fake bias scores
def generate_article():
    article = fake.text(max_nb_chars=1000)
    # Generate scores that sum up to 1.0
    left = round(random.uniform(0, 1), 4)
    right = round(random.uniform(0, 1 - left), 4)
    neutral = round(1.0 - left - right, 4)
    return {
        "article": article,
        "left_bias_score": left,
        "right_bias_score": right,
        "neutral_bias_score": neutral
    }

# Generate 10,000 rows
data = [generate_article() for _ in range(100000)]

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("bias_dataset_final.csv", index=False)

print("✅ Dataset with 10,000 rows saved as 'bias_dataset.csv'")
