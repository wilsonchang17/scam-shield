from gradio_client import Client

class ScamShieldClient:
    """
    A client class for making predictions using the ScamShield API.
    """
    
    def __init__(self):
        self.client = Client("wilsonchang17/scamshield-api")
    
    def predict_message(self, message):

        try:
            result = self.client.predict(
                message=message,
                api_name="/predict"
            )
            print(f"LLM result: {result}")
            return result
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def explain_message(self, message):
        """Calls the /explain endpoint to get a reason for a scam message."""
        try:
            # The result from this API is expected to be a string with bullet points.
            explanation = self.client.predict(
                message=message,
                api_name="/explain"
            )
            print(f"LLM explanation: {explanation}")
            return explanation
        except Exception as e:
            print(f"Error getting explanation: {e}")
            return "An error occurred while generating the explanation."

# Testing
if __name__ == "__main__":
    # This code only runs if the script is executed directly
    scam_shield = ScamShieldClient()
    result = scam_shield.predict_message("We are an investment and loan financing group. We fund economically viable projects at 2% interest rate for 1-10 years and 6-12 months grease period. Ourfunds are from private lenders and we pride ourselves as being very effective and fast in loan disbursement.  I can be reached on email and Whatsapp: +97155 647 4204.  Contact us for more details.  Regards,  MA  Financial Consultant ")
    print(f"Prediction result: {result}")

    # Example for testing the new explain method
    scam_message = "Virginia Department of Transportation Toll Violation Notice: This is your final notice regarding the unpaid toll balance on your account."
    explanation = scam_shield.explain_message(scam_message)
    print(f"Explanation result: {explanation}")