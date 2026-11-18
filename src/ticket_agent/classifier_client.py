import httpx

from ticket_agent.schemas import PredictionRequest, PredictionResponse


class ClassifierClient:
    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http_client = http_client

    async def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        response = await self._http_client.post(
            "/v1/predictions",
            json=request.model_dump(mode="json"),
        )

        response.raise_for_status()

        return PredictionResponse.model_validate(response.json())