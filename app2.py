# app.py - Fixed FastAPI Backend
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os
import json
import google.generativeai as genai
import uvicorn

# Import your existing classes
from datetime import datetime, timedelta
import random
from faker import Faker
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Banking Offer Recommender API")

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Faker for generating realistic data
fake = Faker('en_IN')

# Pydantic models for request/response
class ExistingCustomerRequest(BaseModel):
    customer_name: str

class NewCustomerRequest(BaseModel):
    name: str
    salary: int
    age: int  # MODIFIED: Added age
    banks: List[str]  # MODIFIED: Added list of banks
    primary_interest: str # MODIFIED: Changed from 'interests' for clarity

class CustomerData(BaseModel):
    name: str
    age: int
    salary: int
    recommended_category: str
    expenditure_on_category: float
    banks: List[str]

class OfferData(BaseModel):
    offer_name: str
    bank_name: str
    offer_details_text: str
    eligibility_criteria: str
    is_best_match: bool = False  # Added field to identify best offer

class RecommendationResponse(BaseModel):
    customer: CustomerData
    offers: List[OfferData]
    personalized_message: str

# Your existing classes (copy from your original code)
class SyntheticDataGenerator:
    def __init__(self):
        self.banks = ['HDFC', 'ICICI', 'SBI', 'Axis']
        self.categories = ['groceries', 'shopping', 'fuel', 'travel', 'bills']
        self.salary_ranges = {
            'low': (20000, 40000),
            'medium': (40001, 80000),
            'high': (80001, 150000)
        }
    
    def generate_customers(self, num_customers=1000):
        customers = []
        for i in range(num_customers):
            income_bracket = random.choice(list(self.salary_ranges.keys()))
            min_salary, max_salary = self.salary_ranges[income_bracket]
            salary = random.randint(min_salary, max_salary)
            
            customer = {
                'customer_id': f'CUST_{i+1:06d}',
                'name': fake.name(),
                'age': random.randint(22, 65),
                'city': random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata']),
                'salary': salary,
                'num_accounts': random.randint(1, 4),
            }
            customers.append(customer)
        return pd.DataFrame(customers)
    
    def generate_transactions(self, customers_df, months_history=6):
        transactions = []
        spending_patterns = {
            'low': {'groceries': (2000, 6000), 'shopping': (1000, 4000), 'fuel': (1500, 3000), 'travel': (500, 2000), 'bills': (2000, 5000)},
            'medium': {'groceries': (4000, 12000), 'shopping': (3000, 10000), 'fuel': (3000, 7000), 'travel': (2000, 8000), 'bills': (3000, 8000)},
            'high': {'groceries': (8000, 20000), 'shopping': (8000, 25000), 'fuel': (5000, 12000), 'travel': (5000, 20000), 'bills': (5000, 15000)}
        }

        def get_income_bracket(salary):
            if salary <= 40000: return 'low'
            if salary <= 80000: return 'medium'
            return 'high'
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            income_bracket = get_income_bracket(customer['salary'])
            
            num_accounts = random.randint(1, 4)
            customer_banks = random.sample(self.banks, k=min(num_accounts, len(self.banks)))
            
            for month_offset in range(months_history):
                transaction_date = datetime.now() - timedelta(days=30 * month_offset)
                
                for account_num, bank_name in enumerate(customer_banks):
                    account_id = f"{customer_id}_ACC_{bank_name}_{account_num + 1}"
                    
                    for category in self.categories:
                        min_spend, max_spend = spending_patterns[income_bracket][category]
                        if random.random() > 0.3:
                            monthly_spend = random.randint(min_spend, max_spend)
                            num_transactions = random.randint(1, 8)
                            if num_transactions > 0:
                                amounts = np.random.dirichlet(np.ones(num_transactions)) * monthly_spend
                                for amount in amounts:
                                    if amount > 50:
                                        transaction = {
                                            'transaction_id': f'TXN_{len(transactions)+1:010d}',
                                            'customer_id': customer_id,
                                            'account_id': account_id,
                                            'amount': round(amount, 2),
                                            'category': category,
                                            'transaction_date': transaction_date + timedelta(days=random.randint(0, 29)),
                                            'bank_name': bank_name
                                        }
                                        transactions.append(transaction)
        return pd.DataFrame(transactions)

class DataAggregator:
    def __init__(self, transactions_df):
        self.transactions_df = transactions_df
    
    def create_customer_spending_profiles(self):
        customer_spending = self.transactions_df.groupby(['customer_id', 'category'])['amount'].sum().reset_index()
        spending_profiles = customer_spending.pivot(index='customer_id', columns='category', values='amount').fillna(0)
        return spending_profiles.reset_index()

class FederatedRecommender:
    def __init__(self, banks, categories):
        self.banks = banks
        self.categories = sorted(categories)
        self.global_model = LogisticRegression(solver='liblinear')
        self.scaler = StandardScaler()

    def train_local_models(self, bank_data):
        X = bank_data.drop(columns=['customer_id', 'dominant_category'])
        y = bank_data['dominant_category']
        
        category_cols = ['groceries', 'shopping', 'fuel', 'travel', 'bills']
        X = X[[col for col in category_cols if col in X.columns]]
        
        if X.empty or y.empty:
            return None
        
        X_scaled = self.scaler.fit_transform(X)
        model = LogisticRegression(solver='liblinear', max_iter=200)
        model.fit(X_scaled, y)
        return model

    def federated_train(self, bank_spending_profiles):
        print("Starting Federated Learning rounds...")
        
        padded_coefs = []
        padded_intercepts = []
        
        all_categories = sorted(['groceries', 'shopping', 'fuel', 'travel', 'bills'])
        num_features = len(all_categories)
        
        for bank in self.banks:
            if bank in bank_spending_profiles and not bank_spending_profiles[bank].empty:
                profiles_df = bank_spending_profiles[bank].copy()
                
                for cat in all_categories:
                    if cat not in profiles_df.columns:
                        profiles_df[cat] = 0
                
                profiles_df['dominant_category'] = profiles_df.drop(columns=['customer_id']).idxmax(axis=1)
                
                profiles_df = profiles_df[profiles_df['dominant_category'] != '']
                if profiles_df.empty:
                    continue
                
                local_model = self.train_local_models(profiles_df)
                
                if local_model is not None:
                    local_coefs = local_model.coef_
                    local_intercepts = local_model.intercept_
                    local_classes = list(local_model.classes_)

                    padded_coef = np.zeros((len(all_categories), num_features))
                    padded_intercept = np.zeros(len(all_categories))

                    for i, cls in enumerate(local_classes):
                        global_cls_index = all_categories.index(cls)
                        padded_coef[global_cls_index, :] = local_coefs[i, :]
                        padded_intercept[global_cls_index] = local_intercepts[i]
                    
                    padded_coefs.append(padded_coef)
                    padded_intercepts.append(padded_intercept)
        
        if padded_coefs:
            self.global_model.classes_ = np.array(all_categories)
            self.global_model.coef_ = np.mean(padded_coefs, axis=0)
            self.global_model.intercept_ = np.mean(padded_intercepts, axis=0)
            print("Federated training complete. Global model weights aggregated.")
        else:
            print("No data available to train the federated model.")

    def recommend_category(self, customer_profile):
        profile_df = pd.DataFrame([customer_profile])
        
        category_cols = ['groceries', 'shopping', 'fuel', 'travel', 'bills']
        for col in category_cols:
            if col not in profile_df.columns:
                profile_df[col] = 0
        
        features = profile_df[category_cols]
        features_scaled = self.scaler.transform(features)
        
        predicted_category = self.global_model.predict(features_scaled)[0]
        return predicted_category

def get_eligible_offers(customer_data, offers_db):
    """Retrieves all eligible offers for a customer across all their banks and the recommended category."""
    eligible_offers_list = []
    
    for bank in customer_data['banks']:
        normalized_bank = bank.lower().replace(' ', '')
        
        bank_offers = [
            offer for offer in offers_db
            if offer['bank_name'].lower().replace(' ', '') == normalized_bank
            and offer['offer_type'] == customer_data['recommended_category']
        ]

        eligible_offers = [
            offer for offer in bank_offers
            if customer_data['salary'] >= offer.get('minimum_salary', 0)
            and customer_data['expenditure_on_category'] >= offer.get('minimum_spend', 0)
        ]
        
        eligible_offers_list.extend(eligible_offers)

    return eligible_offers_list

class CentralizedRecommender:
    def __init__(self, weights={'relevance': 0.5, 'value': 0.3, 'priority': 0.2}):
        self.weights = weights
    
    def calculate_benefit(self, offer, customer_expenditure):
        """Calculates the potential monetary benefit of an offer."""
        benefit = 0
        if offer['benefit_type'] == 'discount' and offer['benefit_multiplier']:
            potential_benefit = customer_expenditure * offer['benefit_multiplier']
            benefit = min(potential_benefit, offer.get('maximum_benefit', potential_benefit) if offer.get('maximum_benefit') is not None else potential_benefit)
        elif offer['benefit_type'] == 'cashback' and offer['benefit_multiplier']:
            potential_benefit = customer_expenditure * offer['benefit_multiplier']
            benefit = min(potential_benefit, offer.get('maximum_benefit', potential_benefit) if offer.get('maximum_benefit') is not None else potential_benefit)
        elif offer['benefit_type'] == 'reward_points' and offer['benefit_multiplier']:
            benefit = customer_expenditure * offer['benefit_multiplier'] 
        elif offer['benefit_type'] in ['bogo', 'surcharge_waiver']:
            benefit = offer.get('maximum_benefit', 200)

        return benefit

    def recommend_best_offer(self, all_eligible_offers, customer_data):
        if not all_eligible_offers:
            return None

        for offer in all_eligible_offers:
            relevance = customer_data['expenditure_on_category'] / offer['minimum_spend'] if offer['minimum_spend'] > 0 else 1.0
            offer['relevance_score'] = min(relevance, 2.0)
            offer['benefit_value'] = self.calculate_benefit(offer, customer_data['expenditure_on_category'])

        max_priority = max([o['priority_score'] for o in all_eligible_offers]) if all_eligible_offers else 1
        max_benefit = max([o['benefit_value'] for o in all_eligible_offers]) if all_eligible_offers else 1
        max_relevance = max([o['relevance_score'] for o in all_eligible_offers]) if all_eligible_offers else 1
        
        best_offer = None
        highest_score = -1
        
        for offer in all_eligible_offers:
            normalized_priority = offer['priority_score'] / max_priority
            normalized_benefit = offer['benefit_value'] / max_benefit
            normalized_relevance = offer['relevance_score'] / max_relevance
            
            final_score = (self.weights['relevance'] * normalized_relevance) + \
                          (self.weights['value'] * normalized_benefit) + \
                          (self.weights['priority'] * normalized_priority)

            offer['final_score'] = final_score  # Store the score for debugging
            
            if final_score > highest_score:
                highest_score = final_score
                best_offer = offer
        
        return best_offer

def generate_personalized_message_with_gemini(customer_details, best_offer):
    """Uses the Gemini API to generate a personalized recommendation message."""
    # Replace with your actual API key
    API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    
    # Fallback message function
    def create_fallback_message():
        if not best_offer:
            return f"Hi {customer_details['name']}! We couldn't find a top offer for you at this time, but we're always looking for new deals!"
        
        return f"Hi {customer_details['name']}! Based on your {customer_details['recommended_category']} spending of ₹{customer_details['expenditure_on_category']:,.2f}, we recommend the {best_offer['offer_name']} from {best_offer['bank_name']}. This offer is perfectly tailored to your spending habits and can help you save significantly!"
    
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Gemini API configuration failed: {e}")
        return create_fallback_message()

    if not best_offer:
        return create_fallback_message()

    prompt = f"""
    You are a friendly and helpful bank assistant. Your goal is to create a short, personalized, and exciting recommendation message for a customer based on their spending habits and a specific offer.

    Here are the customer's details:
    - Name: {customer_details['name']}
    - Recommended Spending Category: {customer_details['recommended_category']}
    - Their total spending in this category: ₹{customer_details['expenditure_on_category']:,.2f}
    - Their banks: {', '.join(customer_details['banks'])}

    Here is the best offer we found for them:
    - Offer Name: {best_offer['offer_name']}
    - Bank: {best_offer['bank_name']}
    - Offer Details: {best_offer['offer_details_text']}
    
    Please generate a personalized message that does the following:
    1. Greets the customer by their name.
    2. Acknowledges their interest in their favorite category ('{customer_details['recommended_category']}').
    3. Presents the recommended offer clearly and enthusiastically.
    4. Explains why this specific offer from '{best_offer['bank_name']}' is a great match for them.
    5. Keep the tone positive and encouraging. Do not exceed 80 words.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return create_fallback_message()

# Global variables for loaded data and models
customers_df = None
transactions_df = None
fed_recommender = None
offers_db = None

def initialize_system():
    """Initialize the recommendation system with data and models."""
    global customers_df, transactions_df, fed_recommender, offers_db
    
    customers_path = 'data/synthetic_customers.csv'
    transactions_path = 'data/synthetic_transactions.csv'
    
    print("--- Data Check ---")
    if os.path.exists(customers_path) and os.path.exists(transactions_path):
        print("Existing data files found. Loading data...")
        customers_df = pd.read_csv(customers_path)
        transactions_df = pd.read_csv(transactions_path)
    else:
        print("Data files not found. Generating new synthetic data...")
        data_gen = SyntheticDataGenerator()
        customers_df = data_gen.generate_customers(num_customers=1000)
        transactions_df = data_gen.generate_transactions(customers_df, months_history=6)
        
        os.makedirs('data', exist_ok=True)
        customers_df.to_csv(customers_path, index=False)
        transactions_df.to_csv(transactions_path, index=False)
        print("New synthetic data generated and saved.")

    customers_df['name'] = customers_df['name'].str.strip()
    
    print(f"- Customers: {len(customers_df)}")
    print(f"- Transactions: {len(transactions_df)}")
    
    print("\n--- Training Federated Learning Model ---")
    data_gen = SyntheticDataGenerator()
    bank_spending_profiles = {}
    for bank in data_gen.banks:
        transactions_in_bank = transactions_df[transactions_df['bank_name'] == bank]
        aggregator = DataAggregator(transactions_in_bank)
        profiles = aggregator.create_customer_spending_profiles()
        bank_spending_profiles[bank] = profiles
        
    all_profiles = pd.concat(bank_spending_profiles.values()).drop_duplicates(subset='customer_id')
    
    fed_recommender = FederatedRecommender(data_gen.banks, data_gen.categories)
    
    cols_to_drop = [col for col in all_profiles.columns if col not in data_gen.categories and col != 'customer_id']
    scaler_input = all_profiles.drop(columns=cols_to_drop)
    fed_recommender.scaler.fit(scaler_input.drop(columns=['customer_id']))
    
    fed_recommender.federated_train(bank_spending_profiles)
    
    # Load offers database
    try:
        with open('offers_db.json', 'r') as f:
            offers_db = json.load(f)
        print("Offers database loaded successfully.")
    except FileNotFoundError:
        print("Warning: 'offers_db.json' not found. Using default offers.")
        offers_db = []

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the system when the API starts."""
    initialize_system()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found. Please create templates/index.html</h1>")

@app.get("/api/customers")
async def get_customers():
    """Get list of all customers."""
    if customers_df is not None:
        customer_names = customers_df['name'].unique().tolist()
        return {"customers": customer_names}
    return {"customers": []}

@app.post("/api/existing-customer-recommendation", response_model=RecommendationResponse)
async def get_existing_customer_recommendation(request: ExistingCustomerRequest):
    """Get recommendation for an existing customer."""
    global customers_df, transactions_df, fed_recommender, offers_db
    
    if customers_df is None or transactions_df is None or fed_recommender is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Find customer
    sample_customer_df = customers_df[customers_df['name'] == request.customer_name]
    if sample_customer_df.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{request.customer_name}' not found")

    sample_customer = sample_customer_df.iloc[0]
    sample_transactions = transactions_df[transactions_df['customer_id'] == sample_customer['customer_id']]
    
    # Generate recommendation
    aggregator = DataAggregator(sample_transactions)
    sample_profile_df = aggregator.create_customer_spending_profiles()
    sample_profile = sample_profile_df.iloc[0].to_dict()

    recommended_category = fed_recommender.recommend_category(sample_profile)
    recommended_expenditure = sample_transactions[sample_transactions['category'] == recommended_category]['amount'].sum()
    customer_banks = sorted(sample_transactions['bank_name'].unique().tolist())
    
    customer_data = {
        'name': request.customer_name,
        'recommended_category': recommended_category,
        'salary': sample_customer['salary'],
        'expenditure_on_category': recommended_expenditure,
        'banks': customer_banks
    }
    
    # Get offers and find best one
    all_eligible_offers = get_eligible_offers(customer_data, offers_db)
    centralized_recommender = CentralizedRecommender()
    best_offer = centralized_recommender.recommend_best_offer(all_eligible_offers, customer_data)
    
    # Generate personalized message using the BEST OFFER
    customer_info_for_gemini = {
        'name': request.customer_name,
        'recommended_category': recommended_category,
        'expenditure_on_category': recommended_expenditure,
        'banks': customer_banks
    }
    personalized_message = generate_personalized_message_with_gemini(customer_info_for_gemini, best_offer)
    
    # Format offers for response and sort by best match first
    formatted_offers = []
    
    if best_offer:
        best_offer_id = best_offer.get('offer_id') or f"{best_offer['offer_name']}_{best_offer['bank_name']}"
        sorted_offers = sorted(all_eligible_offers, 
                               key=lambda x: 0 if (x.get('offer_id') or f"{x['offer_name']}_{x['bank_name']}") == best_offer_id else 1)
    else:
        sorted_offers = all_eligible_offers
    
    for i, offer in enumerate(sorted_offers[:5]):  # Limit to top 5 offers
        is_best = (i == 0 and best_offer is not None)
        formatted_offers.append(OfferData(
            offer_name=offer.get('offer_name', 'Special Offer'),
            bank_name=offer.get('bank_name', 'Bank'),
            offer_details_text=offer.get('offer_details_text', 'Great offer for you!'),
            eligibility_criteria=offer.get('eligibility_criteria', 'Standard eligibility applies'),
            is_best_match=is_best
        ))
    
    return RecommendationResponse(
        customer=CustomerData(
            name=request.customer_name,
            age=int(sample_customer['age']),
            salary=int(sample_customer['salary']),
            recommended_category=recommended_category,
            expenditure_on_category=float(recommended_expenditure),
            banks=customer_banks
        ),
        offers=formatted_offers,
        personalized_message=personalized_message
    )

@app.post("/api/new-customer-recommendation", response_model=RecommendationResponse)
async def get_new_customer_recommendation(request: NewCustomerRequest):
    """Get recommendation for a new customer."""
    if not offers_db:
        raise HTTPException(status_code=500, detail="Offers database not available")
    
    # MODIFIED: Use data directly from the request instead of assuming defaults
    primary_category = request.primary_interest
    estimated_spend = request.salary * 0.15 # Assume 15% of salary spent on primary category

    customer_data = {
        'name': request.name,
        'recommended_category': primary_category,
        'salary': request.salary,
        'expenditure_on_category': estimated_spend,
        'banks': request.banks # Use banks from request
    }
    
    # Get offers and find best one
    all_eligible_offers = get_eligible_offers(customer_data, offers_db)
    centralized_recommender = CentralizedRecommender()
    best_offer = centralized_recommender.recommend_best_offer(all_eligible_offers, customer_data)
    
    # Generate personalized message using the BEST OFFER
    customer_info_for_gemini = {
        'name': request.name,
        'recommended_category': primary_category,
        'expenditure_on_category': estimated_spend,
        'banks': request.banks
    }
    personalized_message = generate_personalized_message_with_gemini(customer_info_for_gemini, best_offer)
    
    # Format offers for response and sort by best match first
    formatted_offers = []
    
    if best_offer:
        best_offer_id = best_offer.get('offer_id') or f"{best_offer['offer_name']}_{best_offer['bank_name']}"
        sorted_offers = sorted(all_eligible_offers, 
                               key=lambda x: 0 if (x.get('offer_id') or f"{x['offer_name']}_{x['bank_name']}") == best_offer_id else 1)
    else:
        sorted_offers = all_eligible_offers
    
    for i, offer in enumerate(sorted_offers[:5]):  # Limit to top 5 offers
        is_best = (i == 0 and best_offer is not None)
        formatted_offers.append(OfferData(
            offer_name=offer.get('offer_name', 'Special Offer'),
            bank_name=offer.get('bank_name', 'Bank'),
            offer_details_text=offer.get('offer_details_text', 'Great offer for you!'),
            eligibility_criteria=offer.get('eligibility_criteria', 'Standard eligibility applies'),
            is_best_match=is_best
        ))
    
    return RecommendationResponse(
        customer=CustomerData(
            name=request.name,
            age=request.age, # MODIFIED: Use age from request
            salary=request.salary,
            recommended_category=primary_category,
            expenditure_on_category=estimated_spend,
            banks=request.banks # MODIFIED: Use banks from request
        ),
        offers=formatted_offers,
        personalized_message=personalized_message
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)