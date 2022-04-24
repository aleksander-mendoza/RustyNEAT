from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero import handlers


class MLGWB(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MLGWB-v0'

        super().__init__(*args, max_episode_steps=1000,
                         reward_threshold=100.0,
                         **kwargs)

    def create_server_world_generators(self):
        return [
            handlers.FlatWorldGenerator(generatorString="1;7,2x3,2;1"),
            # generate a 3x3 square of obsidian high in the air and a gold block
            # somewhere below it on the ground
            handlers.DrawingDecorator("""
                <DrawCuboid x1="0" y1="5" z1="-6" x2="0" y2="5" z2="-6" type="gold_block"/>
                <DrawCuboid x1="-2" y1="88" z1="-2" x2="2" y2="88" z2="2" type="obsidian"/>
            """)
        ]

    def create_agent_start(self):
        return [
            # make the agent start with these items
            handlers.SimpleInventoryAgentStart([
                dict(type="water_bucket", quantity=1),
                dict(type="diamond_pickaxe", quantity=1)
            ]),
            # make the agent start 90 blocks high in the air
            handlers.AgentStartPlacement(0, 90, 0, 0, 0)
        ]

    def create_actionables(self):
        return super().create_actionables() + [
            # allow agent to place water
            handlers.KeybasedCommandAction("use"),
            # also allow it to equip the pickaxe
            handlers.EquipAction(["diamond_pickaxe"])
        ]

    def create_observables(self):
        return super().create_observables() + [
            # current location and lifestats are returned as additional
            # observations
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromLifeStats()
        ]

    def create_server_initial_conditions(self):
        return [
            # Sets time to morning and stops passing of time
            handlers.TimeInitialCondition(False, 23000)
        ]
