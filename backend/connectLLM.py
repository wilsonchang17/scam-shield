from gradio_client import Client

class ScamShieldClient:
    """
    A client class for making predictions using the ScamShield API.
    """
    
    def __init__(self):
        """Initialize the client connection to the ScamShield API."""
        self.client = Client("wilsonchang17/scamshield-api")
    
    def predict_message(self, message):
        """
        Send a message to the ScamShield API for prediction.
        
        Args:
            message (str): The message to analyze for scam detection
            
        Returns:
            The prediction result from the API
        """
        try:
            result = self.client.predict(
                message=message,
                api_name="/predict"
            )
            return result
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # This code only runs if the script is executed directly
    scam_shield = ScamShieldClient()
    result = scam_shield.predict_message("We are an investment and loan financing group. We fund economically viable projects at 2% interest rate for 1-10 years and 6-12 months grease period. Ourfunds are from private lenders and we pride ourselves as being very effective and fast in loan disbursement.  I can be reached on email and Whatsapp: +97155 647 4204.  Contact us for more details.  Regards,  MA  Financial Consultant ")
    print(result)