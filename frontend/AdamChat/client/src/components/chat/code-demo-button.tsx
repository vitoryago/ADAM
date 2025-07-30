import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Code2, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CodeDemoButtonProps {
  onSendMessage: (message: string) => void;
  className?: string;
}

export function CodeDemoButton({ onSendMessage, className }: CodeDemoButtonProps) {
  const [isLoading, setIsLoading] = useState(false);

  const handleDemoClick = async () => {
    setIsLoading(true);
    
    const demoMessage = `Can you show me some advanced code examples with syntax highlighting? Here are a few different languages:

**JavaScript/React Component:**
\`\`\`jsx
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';

const UserProfile = ({ userId }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchUser(userId).then(userData => {
      setUser(userData);
      setLoading(false);
    });
  }, [userId]);
  
  if (loading) return <div>Loading...</div>;
  
  return (
    <div className="profile-card">
      <h2>{user?.name}</h2>
      <p>Email: {user?.email}</p>
      <Button onClick={() => handleEdit(user.id)}>
        Edit Profile
      </Button>
    </div>
  );
};
\`\`\`

**Python Data Processing:**
\`\`\`python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def process_dataset(file_path):
    # Load and clean data
    df = pd.read_csv(file_path)
    df = df.dropna()
    
    # Feature engineering
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Usage
model, test_X, test_y = process_dataset('data.csv')
accuracy = model.score(test_X, test_y)
print(f"Model accuracy: {accuracy:.2%}")
\`\`\`

**SQL Database Query:**
\`\`\`sql
SELECT 
    u.id,
    u.name,
    u.email,
    COUNT(o.id) as order_count,
    SUM(o.total_amount) as total_spent,
    AVG(o.total_amount) as avg_order_value
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2024-01-01'
    AND u.status = 'active'
GROUP BY u.id, u.name, u.email
HAVING COUNT(o.id) > 0
ORDER BY total_spent DESC
LIMIT 10;
\`\`\`

**CSS Animation:**
\`\`\`css
.floating-button {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.floating-button:hover {
  transform: translateY(-3px) scale(1.05);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

.floating-button::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 2px;
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  mask: linear-gradient(#fff 0 0) content-box, 
        linear-gradient(#fff 0 0);
  mask-composite: exclude;
  animation: pulse 2s infinite;
}
\`\`\`

Also test inline code like \`const API_URL = 'https://api.example.com'\` and \`npm install react\` and \`git commit -m "Add new feature"\` to see how they look!`;

    onSendMessage(demoMessage);
    
    // Reset after a brief delay
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  };

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={handleDemoClick}
      disabled={isLoading}
      className={cn(
        "relative group overflow-hidden transition-all duration-300",
        "border-dashed border-blue-300 dark:border-blue-700",
        "hover:border-solid hover:border-blue-500 dark:hover:border-blue-400",
        "hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50",
        "dark:hover:from-blue-950/50 dark:hover:to-purple-950/50",
        isLoading && "animate-pulse",
        className
      )}
    >
      <div className="flex items-center gap-2">
        {isLoading ? (
          <Sparkles className="w-4 h-4 animate-spin text-blue-600 dark:text-blue-400" />
        ) : (
          <Code2 className="w-4 h-4 text-blue-600 dark:text-blue-400 group-hover:rotate-12 transition-transform duration-300" />
        )}
        <span className="text-blue-700 dark:text-blue-300 font-medium">
          {isLoading ? 'Sending...' : 'Demo Code Highlighting'}
        </span>
      </div>
      
      {/* Animated background effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 to-purple-400/20 translate-x-full group-hover:translate-x-0 transition-transform duration-500 ease-out" />
    </Button>
  );
}