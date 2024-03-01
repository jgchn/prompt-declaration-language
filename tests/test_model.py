from pdl.pdl.pdl_ast import Program  # pyright: ignore
from pdl.pdl.pdl_interpreter import empty_scope  # pyright: ignore
from pdl.pdl.pdl_interpreter import process_block  # pyright: ignore

model_data = {
    "description": "Hello world with a variable to call into a model",
    "prompts": [
        "Hello,",
        {
            "model": "ibm/granite-20b-code-instruct-v1",
            "parameters": {
                "decoding_method": "greedy",
                "stop_sequences": ["!"],
                "include_stop_sequence": False,
            },
        },
        "!\n",
    ],
}


def test_model():
    log = []
    data = Program.model_validate(model_data)
    _, document, _, _ = process_block(log, empty_scope, data.root)
    assert document == "Hello, world!\n"


model_chain_data = {
    "description": "Hello world showing model chaining",
    "prompts": [
        "Hello,",
        {
            "def": "SOMEONE",
            "prompts": [
                {
                    "model": "ibm/granite-20b-code-instruct-v1",
                    "parameters": {
                        "decoding_method": "greedy",
                        "stop_sequences": ["!"],
                        "include_stop_sequence": False,
                    },
                }
            ],
        },
        "!\n",
        "Who is",
        {"get": "SOMEONE"},
        "?\n",
        {
            "def": "RESULT",
            "prompts": [
                {
                    "model": "google/flan-t5-xl",
                    "parameters": {
                        "decoding_method": "greedy",
                        "stop_sequences": ["!"],
                        "include_stop_sequence": False,
                    },
                }
            ],
        },
        "\n",
    ],
}


def test_model_chain():
    log = []
    data = Program.model_validate(model_chain_data)
    _, document, _, _ = process_block(log, empty_scope, data.root)
    assert document == "".join(
        [
            "Hello,",
            " world",
            "!\n",
            "Who is",
            " world",
            "?\n",
            "hello world",
            "\n",
        ]
    )


multi_shot_data = {
    "description": "Hello world showing model chaining",
    "prompts": [
        {
            "def": "LOCATION",
            "prompts": [
                {
                    "model": "ibm/granite-20b-code-instruct-v1",
                    "input": {
                        "prompts": [
                            "Question: What is the weather in London?\n",
                            "London\n",
                            "Question: What's the weather in Paris?\n",
                            "Paris\n",
                            "Question: Tell me the weather in Lagos?\n",
                            "Lagos\n",
                            "Question: What is the weather in Armonk, NY?\n",
                        ]
                    },
                    "parameters": {
                        "decoding_method": "greedy",
                        "stop_sequences": ["Question"],
                        "include_stop_sequence": False,
                    },
                }
            ],
            "show_result": True,
        }
    ],
}


def test_multi_shot():
    log = []
    data = Program.model_validate(multi_shot_data)
    _, document, _, _ = process_block(log, empty_scope, data.root)
    assert document == "Armonk, NY\n"


model_data_missing_parameters = {
    "description": "Hello world with a variable to call into a model",
    "prompts": [
        "Hello,\n",
        {
            "model": "ibm/granite-20b-code-instruct-v1",
            "parameters": {
                "stop_sequences": ["."],
            },
        },
    ],
}


def test_data_missing_parameters():
    log = []
    data = Program.model_validate(model_data_missing_parameters)
    _, document, _, _ = process_block(log, empty_scope, data.root)
    assert document == "Hello,\n\nI am a student at the University of Toronto."
