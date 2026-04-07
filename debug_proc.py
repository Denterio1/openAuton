
import sys, asyncio
sys.path.insert(0, 'src')
from llm.provider import ModelProvider
from experience.processor import ReflectionProcessor
from experience.episodes import ExperienceEpisode, TaskType, EpisodeStatus, EvaluationResult

llm = ModelProvider.from_env()
p = ReflectionProcessor(llm_provider=llm)

ep = ExperienceEpisode.start("Train a transformer", TaskType.TRAIN_FROM_SCRATCH)
ep.finish(EpisodeStatus.PARTIAL, EvaluationResult(accuracy=0.72))

loop = asyncio.new_event_loop()
report = loop.run_until_complete(p.process_async(ep, history=[]))
loop.close()

print(report.priority_actions)
print(report.lessons_learned)