import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

DEPT_BASE = {  # base annual salary (INR) for a fresher
    "Technical Trainers": 300000, "Finance & Operations": 280000,
    "Business Development Executive": 250000, "HR": 260000,
    "Placement Coordinator": 240000, "Sales & Marketing": 255000,
}
DEPT_RATE = {  # yearly raise per year of experience
    "Technical Trainers": 55000, "Finance & Operations": 45000,
    "Business Development Executive": 42000, "HR": 38000,
    "Placement Coordinator": 34000, "Sales & Marketing": 44000,
}
DEPT_SKILLS = {
    "Technical Trainers": ["Python", "Java", "SQL", "Machine Learning", "Web Development", "Communication"],
    "Finance & Operations": ["Excel", "Accounting", "Tally", "SQL", "Communication"],
    "Business Development Executive": ["Sales", "CRM", "Negotiation", "Communication", "Excel"],
    "HR": ["Recruitment", "Payroll", "Communication", "Excel", "CRM"],
    "Placement Coordinator": ["Recruitment", "Networking", "CRM", "Communication", "Excel"],
    "Sales & Marketing": ["Sales", "Digital Marketing", "SEO", "CRM", "Communication"],
}
SKILL_BONUS = {"Machine Learning": 60000, "Python": 30000, "Java": 25000, "SQL": 20000,
               "Web Development": 25000, "Digital Marketing": 20000, "SEO": 12000,
               "Negotiation": 18000, "Accounting": 20000, "Tally": 10000, "Sales": 15000,
               "CRM": 10000, "Recruitment": 12000, "Payroll": 10000, "Networking": 10000,
               "Excel": 8000, "Communication": 8000}
EDU_BONUS = {"Diploma": 0, "Bachelors": 40000, "Masters": 100000, "MBA": 130000, "PhD": 180000}


def generate(n=200):
    rows = []
    depts = list(DEPT_BASE)
    for _ in range(n):
        dept = rng.choice(depts, p=[.22, .16, .18, .12, .12, .20])
        edu = rng.choice(list(EDU_BONUS), p=[.08, .50, .24, .15, .03])
        exp = float(np.clip(rng.gamma(2.2, 2.6), 0, 25).round(1))
        age = int(np.clip(21 + exp + rng.integers(0, 6), 20, 60))
        k = rng.integers(1, min(5, len(DEPT_SKILLS[dept])) + 1)
        skills = list(rng.choice(DEPT_SKILLS[dept], size=k, replace=False))
        salary = (DEPT_BASE[dept] + DEPT_RATE[dept] * exp ** 0.92 + EDU_BONUS[edu]
                  + sum(SKILL_BONUS[s] for s in skills))
        salary *= rng.normal(1, 0.06)
        rows.append([exp, edu, ";".join(skills), dept, age, int(round(salary, -3))])
    return pd.DataFrame(rows, columns=["Experience", "Education", "Skills", "Department", "Age", "Salary"])


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/employee_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/employee_data.csv")
