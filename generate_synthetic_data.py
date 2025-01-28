import os
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from initialize_llm import initialize_llm
from typing_extensions import Annotated, TypedDict

# Function to generate synthetic data
def generate_data():
    # Initialize the language model
    llm = initialize_llm("openai", "gpt-4o")
    
    # Define the system message for the LLM
    system_message = f"""Generate a dataset with give number of  entries. 
                        Each entry should have a 'grievance_text' and a corresponding 'department'. 
                        The 'grievance_text' should describe a common civic issue, 
                        and the 'department' should be the relevant municipal department that handles such issues.
                        The grievance text should belong to one of the following departments:
                        - Water Supply
                        - Sanitation
                        - Roads and Bridges
                        - Public Health

                        Provide the output in the following format:
                        
                        grievance_text: ['issue 1', 'issue 2', ...]
                        department: ['department 1', 'department 2', ...]"""

    # Create a chat prompt template
    action_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])

    # Define the structured output schema
    class StructuredOutput(TypedDict):
        """
        Schema for the structured output of the LLM
        """
        grievance_text: Annotated[list[str], ..., "the grievance text"]
        department: Annotated[list[str], ..., "the corresponding department"]
        
    # Configure the LLM to produce structured output
    structured_llm = llm.with_structured_output(StructuredOutput, include_raw=True)

    # Create a chain of the prompt template and the structured LLM
    chain = action_template | structured_llm

    # Define the prompt to generate data
    prompt = "Give me a dataset with 10 entries, each having a 'grievance_text' and a corresponding 'department'."
    
    # Invoke the chain with the prompt
    response = chain.invoke({
            "input": prompt
                })

    # Return the parsed response
    return response['parsed']

# Initialize the final data dictionary
final_data = {'grievance_text': [], 'department': []}

# Generate data until we have 100 entries
while len(final_data.get('grievance_text')) < 100:
    data = generate_data()
    final_data["grievance_text"].extend(data.get('grievance_text'))
    final_data["department"].extend(data.get('department'))
    print("Data collected so far: ", len(final_data.get('grievance_text')))

# Create a DataFrame from the final data
df = pd.DataFrame(final_data)

# Define the output CSV path
output_csv_path = os.path.join(os.path.dirname(__file__), 'synthetic_data.csv')

# Save the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

# Print a success message
print(f"Data has been successfully saved to {output_csv_path}")
