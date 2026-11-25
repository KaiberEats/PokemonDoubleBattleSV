import logging
import unittest
from types import SimpleNamespace as NS

from poke_env.environment.double_battle import DoubleBattle
from training_errors import TrainingDiagnosticsMixin


class RequestSlotTests(unittest.IsolatedAsyncioTestCase):
    def make_battle(self, role):
        b = DoubleBattle('battle-test-1', 'learner', logging.getLogger('slots'), 9)
        def entry(name, active):
            return dict(ident=f'{role}: {name}', details=name+', L50', condition='100/100',
                        active=active, stats={s: 100 for s in ('atk','def','spa','spd','spe')},
                        moves=['protect'], baseAbility='static', item='')
        # Seed a cached live outgoing Pokemon; the request's incoming Pokemon
        # is also alive, so poke-env's fainted-slot repair does not apply.
        b.parse_request(dict(rqid=1, active=[{'moves':[]},{'moves':[]}],
                             side=dict(id=role, name='learner', pokemon=[entry('Pikachu',True),entry('Raichu',True),entry('Eevee',False)])))
        b.parse_request(dict(rqid=2, active=[{'moves':[dict(move='Protect',id='protect',pp=16,maxpp=16,target='self',disabled=False)]}]*2,
                             side=dict(id=role,name='learner',pokemon=[entry('Pikachu',True),entry('Eevee',True),entry('Raichu',False)])))
        return b

    def test_real_parser_stale_live_slot_is_repaired_for_both_roles(self):
        for role in ('p1','p2'):
            b = self.make_battle(role)
            self.assertIsNone(b.active_pokemon[1])
            TrainingDiagnosticsMixin._sync_active_request(b)
            self.assertEqual(b.active_pokemon[1].species, 'eevee')
            self.assertEqual(b.available_moves[1][0].id, 'protect')
            self.assertEqual(len(b.opponent_active_pokemon), 2)

    def test_replacement_request_keeps_existing_slots(self):
        b = self.make_battle('p1')
        before = dict(b._active_pokemon)
        b._last_request = dict(forceSwitch=[False,True],side=b._last_request['side'])
        TrainingDiagnosticsMixin._sync_active_request(b)
        self.assertEqual(b._active_pokemon, before)

    async def test_wait_request_does_not_send(self):
        class Base:
            async def _handle_battle_request(self, *args, **kwargs):
                raise AssertionError('must not send during wait')
        class Player(TrainingDiagnosticsMixin, Base):
            pass
        player = Player()
        b = NS(_last_request={'wait':True})
        await player._handle_battle_request(b)


if __name__ == '__main__':
    unittest.main()
