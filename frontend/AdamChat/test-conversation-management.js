#!/usr/bin/env node

/**
 * Test script for conversation management features
 * Run with: node test-conversation-management.js
 */

const API_URL = 'http://localhost:8000/api';

async function testAPI(endpoint, method = 'GET', body = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  if (body) {
    options.body = JSON.stringify(body);
  }
  
  try {
    const response = await fetch(`${API_URL}${endpoint}`, options);
    const data = await response.json();
    
    if (!response.ok) {
      console.error(`❌ ${method} ${endpoint} failed:`, response.status, data);
      return null;
    }
    
    console.log(`✅ ${method} ${endpoint} successful`);
    return data;
  } catch (error) {
    console.error(`❌ ${method} ${endpoint} error:`, error.message);
    return null;
  }
}

async function runTests() {
  console.log('🧪 Testing Conversation Management Features\n');
  
  // 1. Get or create a test project
  console.log('1️⃣ Getting projects...');
  let projects = await testAPI('/projects');
  
  let testProject;
  if (!projects || projects.length === 0) {
    console.log('   Creating test project...');
    testProject = await testAPI('/projects', 'POST', {
      name: 'Test Project',
      description: 'Project for testing conversation management'
    });
  } else {
    testProject = projects[0];
    console.log(`   Using existing project: ${testProject.name}`);
  }
  
  if (!testProject) {
    console.error('❌ Could not get or create test project');
    return;
  }
  
  // 2. Create a test conversation
  console.log('\n2️⃣ Creating test conversation...');
  const newConversation = await testAPI(`/projects/${testProject.id}/conversations`, 'POST', {
    title: 'Test Conversation ' + new Date().toISOString()
  });
  
  if (!newConversation) {
    console.error('❌ Could not create conversation');
    return;
  }
  console.log(`   Created: ${newConversation.title} (ID: ${newConversation.id})`);
  
  // 3. Test renaming conversation
  console.log('\n3️⃣ Testing conversation rename...');
  const renamedConversation = await testAPI(`/conversations/${newConversation.id}`, 'PATCH', {
    title: 'Renamed Test Conversation'
  });
  
  if (renamedConversation) {
    console.log(`   Renamed to: ${renamedConversation.title}`);
  }
  
  // 4. Get conversations list
  console.log('\n4️⃣ Getting conversations list...');
  const conversations = await testAPI(`/projects/${testProject.id}/conversations`);
  
  if (conversations) {
    console.log(`   Found ${conversations.length} conversations:`);
    conversations.slice(0, 3).forEach(conv => {
      console.log(`     - ${conv.title} (ID: ${conv.id})`);
    });
  }
  
  // 5. Test deleting conversation
  console.log('\n5️⃣ Testing conversation delete...');
  const deleteResponse = await fetch(`${API_URL}/conversations/${newConversation.id}`, {
    method: 'DELETE'
  });
  
  if (deleteResponse.ok) {
    console.log('   ✅ Successfully deleted test conversation');
  } else {
    console.log('   ❌ Failed to delete conversation');
  }
  
  // 6. Verify deletion
  console.log('\n6️⃣ Verifying deletion...');
  const updatedConversations = await testAPI(`/projects/${testProject.id}/conversations`);
  
  if (updatedConversations) {
    const stillExists = updatedConversations.some(c => c.id === newConversation.id);
    if (!stillExists) {
      console.log('   ✅ Conversation successfully removed from list');
    } else {
      console.log('   ❌ Conversation still exists after deletion');
    }
  }
  
  console.log('\n✨ Testing complete!');
  console.log('\nTo test in the UI:');
  console.log('1. Start the backend: cd src/adam_v2 && python main.py');
  console.log('2. Start the frontend: cd frontend/AdamChat && npm run dev');
  console.log('3. Open http://localhost:5173 in your browser');
  console.log('4. Create or select a project');
  console.log('5. Hover over conversations in the sidebar to see edit/delete buttons');
  console.log('6. Click the edit button to rename a conversation');
  console.log('7. Click the delete button to remove a conversation');
}

// Run tests
runTests().catch(console.error);