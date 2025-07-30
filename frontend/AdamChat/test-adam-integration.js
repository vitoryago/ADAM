// Quick test of ADAM integration
const { getADAMBridge } = require('./server/adam-integration.ts');

async function testADAM() {
  console.log('Testing ADAM integration...');
  
  try {
    const bridge = getADAMBridge();
    await bridge.initialize();
    
    console.log('✅ ADAM bridge initialized');
    
    // Test a simple query
    const response = await bridge.processQuery({
      query: "Hello ADAM, are you working?",
      conversationId: "test-conv-1",
      projectId: "test-project-1", 
      userId: "test-user-1"
    });
    
    console.log('✅ ADAM Response:', response);
    
    await bridge.shutdown();
    console.log('✅ Test completed successfully');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

testADAM();