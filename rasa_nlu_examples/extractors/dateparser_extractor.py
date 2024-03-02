import re
import logging
import datetime as dt
import pytz
from datetime import datetime
from typing import Any, Text, Dict, List, Type
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)
from dateparser.search import search_dates
from parsivar import Normalizer
logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class DateparserEntityExtractor(EntityExtractorMixin, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return []

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["dateparser"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "prefer_dates_from": None,
            "relative_base": None,
            "languages": None,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.entity_name = config.get("entity_name")
        self.settings = {}
        # Dateparser is picky about the dictionary it receives, when value = None,
        # there should be no entry in the dictionary.
        if config.get("prefer_dates_from"):
            self.settings["PREFER_DATES_FROM"] = config.get("prefer_dates_from")
        if config.get("relative_base"):
            base = config.get("relative_base")
            self.settings["RELATIVE_BASE"] = dt.datetime.strptime(base, "%Y-%m-%d %H:%M")
        self.languages = config.get("languages")

    def train(self, training_data: TrainingData) -> Resource:
        pass

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)

    def process(self, messages: List[Message]) -> List[Message]:
      """
      Preprocess text with Parsivar package (just Normalizing).
      example: input >> "یازده شهریور هزار و سیصد و نود و شش"
      output >> "۱۳۹۶/۶/۱۱"
      """
        for message in messages:
            if 'text' in message.data.keys():
                msg = message.data['text']
                message.data['text'] = Normalizer(date_normalizing_needed=True).normalize(msg)
                # Normalize twice to support more complicated sentences
                message.data['text'] = Normalizer(date_normalizing_needed=True).normalize(message.data['text'])
            self._set_entities(message)
        return messages

    def _set_entities(self, message: Message, **kwargs: Any) -> None:
        matches = self._extract_entities(message)
        message.set(ENTITIES, message.get(ENTITIES, []) + matches, add_to_output=True)

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        hits = search_dates(
            message.get(TEXT),
            languages=self.languages if self.languages else None,
            # set relative_base time to the current time
            settings = {"RELATIVE_BASE": datetime.now(pytz.timezone('Asia/Tehran'))},
        )
        if not hits:
            return []

        matches = []
        for substr, timestamp in hits:
            for match in re.finditer(substr, message.get(TEXT)):
                matches.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: "datetime_reference",
                        ENTITY_ATTRIBUTE_START: match.start(),
                        ENTITY_ATTRIBUTE_END: match.end(),
                        ENTITY_ATTRIBUTE_VALUE: message.get(TEXT)[
                            match.start() : match.end()
                        ],
                        "parsed_date": str(timestamp),
                        "confidence": 1.0,
                        "extractor": "DateparserEntityExtractor",
                    }
                )
        return matches

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
