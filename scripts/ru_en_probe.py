"""
Рукописные английские формулировки для десяти вопросов ru_no_anchor.

ЗАЧЕМ. Замер пользы перевода запроса перед поиском: корпус английский,
sparse-ветвь на кириллице молчит (прочерк у 8 из 10). Вопрос заказчика —
не переводить ли запрос. Отвечать надо числами.

ПОЧЕМУ РУКАМИ, А НЕ МОДЕЛЬЮ. Перевод моделью — недетерминированный шаг:
две прогонки дадут две разные строки, и замер перестанет быть точным.
Здесь строки зафиксированы, поэтому весь конвейер остаётся воспроизводимым.

ГРАНИЦА ЧЕСТНОСТИ ЭТОГО ЗАМЕРА, назвать её обязательно.
Русские вопросы писал тот же автор, что и эти переводы, ПО ТЕКСТУ целевых
чанков — иначе вопрос был бы неотвечаем. Полная слепота поэтому невозможна:
нельзя развидеть прочитанное. Смягчение — перевод сделан С РУССКОГО ВОПРОСА,
а не из чанка: где русская формулировка уже несла термин («сдвоенный
масляный фильтр»), английская несёт его же; где не несла — в перевод
не добавлено ничего, чего не было в вопросе. Ни одна английская строка
не сверялась с текстом чанка на предмет совпадения слов.

Из этого следует, как читать результат: измеренная польза перевода —
это ВЕРХНЯЯ оценка. Живой механик формулирует хуже переводчика, знающего
предметную область, и настоящий выигрыш будет не больше измеренного.
"""

EN_PROBE: dict[str, str] = {
    "gs033": "What protective gear should I put on before opening the air tank drain valve?",
    "gs034": "What should I use to clean the lubricator bowl on the air starting motor?",
    "gs035": "What position should the duplex oil filter lever handle be left in after inspection?",
    "gs036": "How long must I wait after stopping the engine before working on the high pressure fuel lines?",
    "gs037": "At what differential pressure should the auxiliary water pump be inspected more often?",
    "gs038": "What should I coat the threads of the speed timing sensor with before installing it?",
    "gs039": "How many spare air cleaner elements are recommended to keep on board the vessel?",
    "gs040": "What should be done with the sea water valve before removing and cleaning the strainer?",
    "gs041": "How often should the generator winding insulation resistance be tested in a humid environment?",
    "gs042": "Should I keep adding grease to the generator bearing until it purges?",
}
