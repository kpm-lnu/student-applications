{
    "compartments": [
        {
            "id": "2278ca91-d735-5a4e-8b6f-c7b8d8d6101c",
            "name": "S",
            "value": 999
        },
        {
            "id": "32bb0d46-06e5-557c-bcc1-d223bfa8ad40",
            "name": "I",
            "value": 1
        },
        {
            "id": "6a8c1f15-3099-505f-bc41-ceb6cfb8ea08",
            "name": "R",
            "value": 0
        }
    ],
    "interventions": [],
    "constants": [
        {
            "id": "396f886b-1f36-5559-a7c8-538871904522",
            "name": "beta",
            "value": 0.05
        },
        {
            "id": "66c33679-9319-505b-9fda-6a54ac5610c7",
            "name": "gamma",
            "value": 0.01
        },
        {
            "id": "de1c200b-ae2f-5eed-88a5-980b5f50a398",
            "name": "N",
            "value": 1000
        }
    ],
    "flows": [
        {
            "id": "4cc56a0d-7ffc-5efa-9004-6f801cb296dc",
            "equation": "beta * S * I / N",
            "source": "2278ca91-d735-5a4e-8b6f-c7b8d8d6101c",
            "target": "32bb0d46-06e5-557c-bcc1-d223bfa8ad40"
        },
        {
            "id": "d4d49a15-196e-5761-9ebc-170c973020b9",
            "equation": "gamma * I",
            "source": "32bb0d46-06e5-557c-bcc1-d223bfa8ad40",
            "target": "6a8c1f15-3099-505f-bc41-ceb6cfb8ea08"
        }
    ]
}
